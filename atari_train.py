"""Train Atari games using CompressionLab, NonLinearLab, and ShapingLab with JAX integration."""

import logging
import os
import pickle
import hydra
import csv
import jax
import json
import numpy as np
import chex
import time
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Optional
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd

import gymnasium as gym
import ale_py
import cma
import jax.numpy as jnp
from compressionlab.api import get_compressor, get_default_config
from NonLinearLab.nonlinearlab.sparsification import Sparsifier
from ShapingLab.shapinglab.shaping import ShapingMapper

@dataclass
class TrainConfig:
    checkpoint: str = "saves/checkpoint"
    resume: bool = False

@dataclass
class CMAConfig:
    tolflatfitness: int = 50
    popsize_factor: int = 100
    sigma0: float = 1.0

@dataclass
class CompressConfig:
    # Compression parameters
    type: str = "dct"              # Options: "dct", "dft", "wavelet", "conv", "aed"
    k: int = 75                    # Size of compressed representation (for dct/dft)
    channel_last: bool = True      # Image format
    norm: str = "ortho"           # Normalization type for transforms
    
    # AED (Autoencoder) specific parameters
    ckpt_path: Optional[str] = field(default=None)  # Path to AED checkpoint if using pretrained
    encoder_features: List[int] = field(default_factory=lambda: [32, 64])
    encoder_kernels: List[int] = field(default_factory=lambda: [3, 3])
    encoder_strides: List[int] = field(default_factory=lambda: [2, 2])
    decoder_features: List[int] = field(default_factory=lambda: [64, 32, 1])
    decoder_kernels: List[int] = field(default_factory=lambda: [3, 3, 3])
    decoder_strides: List[int] = field(default_factory=lambda: [2, 2, 1])
    
    # Image dimensions (mainly for AED)
    batch: int = 1
    height: int = 210  # Atari default
    width: int = 160   # Atari default
    channels: int = 1  # Grayscale
    
    # Sparsification parameters
    p: float = 0.25    # Sparsification ratio

@dataclass
class Config:
    chunk: int = 100
    render: bool = False
    game: str = "ALE/SpaceInvaders-v5"
    steps: int = 1000  # Number of steps per episode
    max_episode_steps: int = 10000
    frameskip: int = 4
    repeat_action_prob: float = 0.0
    comp_cfg: CompressConfig = field(default_factory=CompressConfig)
    cma_cfg: CMAConfig = field(default_factory=CMAConfig)
    train_cfg: TrainConfig = field(default_factory=TrainConfig)
    min_gen: int = 40

cs = ConfigStore.instance()
cs.store(name="config", node=Config)

def save_results(results, results_path):
    results_dir = os.path.dirname(results_path)
    os.makedirs(results_dir, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

def forward(rng, obsv, comp_jit, sparsifier_jit, mapper_jit):
    """Forward pass combining compression, sparsification and mapping."""
    compressed = comp_jit(obsv)
    sparse = sparsifier_jit(compressed)
    logits = mapper_jit(sparse)
    return jnp.squeeze(logits)

def shaper_input_size(cc: CompressConfig):
    """Calculate input dimensions for shaper based on compression config."""
    if cc.type == "dct" or cc.type == "dft":
        M = cc.k  # input rows
        N = cc.k  # input cols
    else:
        raise NotImplementedError(f"Compression type {cc.type} not implemented")
    return (M, N)

def shaper_theta_size(cfg: Config, actionspace):
    """Calculate total parameter size needed for the shaper."""
    input_size = shaper_input_size(cfg.comp_cfg)
    channels = 1  # Grayscale input
    return actionspace * input_size[0] + input_size[1] * channels + actionspace

def get_processing_chain_jit(shaper_theta, cfg: Config, actionspace):
    """Create and JIT-compile the processing chain (compression -> sparsification -> mapping)."""
    cc = cfg.comp_cfg
    input_size = shaper_input_size(cc)
    
    # Setup compression based on type
    if cc.type in ["dct", "dft"]:
        comp_config = {
            "type": cc.type,
            "k": cc.k,
            "channel_last": cc.channel_last,
            "norm": cc.norm,
            "image_height": None,  # Will be inferred from input
            "image_width": None,   # Will be inferred from input
            "image_channels": cc.channels
        }
    elif cc.type == "aed":
        comp_config = {
            "type": "aed",
            "ckpt_path": cc.ckpt_path,
            "batch": cc.batch,
            "height": cc.height,
            "width": cc.width,
            "channels": cc.channels,
            "encoder_features": cc.encoder_features,
            "encoder_kernels": cc.encoder_kernels,
            "encoder_strides": cc.encoder_strides,
            "decoder_features": cc.decoder_features,
            "decoder_kernels": cc.decoder_kernels,
            "decoder_strides": cc.decoder_strides
        }
    elif cc.type == "wavelet":
        comp_config = {
            "type": "wavelet",
            "channel_last": cc.channel_last,
        }
    elif cc.type == "conv":
        comp_config = {
            "type": "conv",
            "channel_last": cc.channel_last,
        }
    else:
        # Fallback to default config
        comp_config = get_default_config()
        
    comp = get_compressor(comp_config)
    
    # Setup sparsification
    sparsifier = Sparsifier(q=cc.p)
    
    # Setup mapping
    meta = {
        "type": "ShapingMapper",
        "shapes": [
            list((actionspace, input_size[0])), 
            list((input_size[1], 1, 1)),  # Using 1 channel for grayscale
            list((actionspace, 1))
        ],
    }
    mapper = ShapingMapper.set_params_jitable(shaper_theta, meta)
    
    return (
        jax.jit(lambda x: comp(x)),
        jax.jit(lambda x: sparsifier(x)),
        jax.jit(lambda x: mapper(x))
    )

def make_env(cfg: Config) -> gym.Env:
    """Create Atari environment with specified configuration."""
    return gym.make(
        id=cfg.game,
        obs_type="grayscale",
        frameskip=cfg.frameskip,
        repeat_action_probability=cfg.repeat_action_prob
    )

@hydra.main(config_name="config", version_base="1.3")
def train(cfg: Config):
    """Main training loop using CMA-ES."""
    logging.getLogger().setLevel(logging.WARNING)
    
    # Verify single device
    devices = jax.devices()
    assert len(devices) == 1, f'Expected single device, found {len(devices)}: {devices}'
    
    rng = jax.random.PRNGKey(42)
    env = make_env(cfg)
    
    # Prepare CSV logging per combination: {compression}_{nonlinearity}_{mapping}_data.csv
    combo_name = f"{cfg.comp_cfg.type}_sparsifier_ShapingMapper"
    data_dir = os.path.join(get_original_cwd(), "data")
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, f"{combo_name}_data.csv")
    init_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if init_header:
        csv_writer.writerow(["generation", "individual", "score"])
        csv_file.flush()
    
    # Initialize CMA-ES
    rng, _rng = jax.random.split(rng)
    if cfg.train_cfg.resume:
        with open(cfg.train_cfg.checkpoint, 'rb') as f:
            es = pickle.load(f)
    else:
        x0 = [0] * shaper_theta_size(cfg, env.action_space.n)
        cma_options = {'popsize_factor': cfg.cma_cfg.popsize_factor,
                      'tolflatfitness': cfg.cma_cfg.tolflatfitness}
        es = cma.CMAEvolutionStrategy(x0, cfg.cma_cfg.sigma0, inopts=cma_options)

    def evaluate_population(sols, rng):
        """Evaluate a population of solutions sequentially."""
        losses = []
        
        for solution_idx, theta in enumerate(sols):
            # Set up policy components
            comp_jit, sparsifier_jit, mapper_jit = get_processing_chain_jit(
                theta, cfg, env.action_space.n
            )

            # Run episode
            total_reward = 0.0
            done = False
            obs, _ = env.reset()
            
            rng, episode_rng = jax.random.split(rng)
            
            for step in range(cfg.steps):
                if done:
                    break
                
                # Convert to float and add batch/channel dims if needed
                obs_proc = obs.astype(np.float32)
                if obs_proc.ndim == 2:
                    obs_proc = obs_proc[None, ..., None]  # Add batch and channel dims
                elif obs_proc.ndim == 3:
                    obs_proc = obs_proc[None]  # Add batch dim
                
                # Process observation through chain
                compressed = np.array(comp_jit(obs_proc))
                sparse = np.array(sparsifier_jit(compressed))
                logits = np.array(mapper_jit(sparse))
                logits = np.squeeze(logits)
                
                # Sample action
                episode_rng, subrng = jax.random.split(episode_rng)
                action = int(jax.random.categorical(subrng, logits).item())
                
                # Step environment
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
            
            loss = -total_reward  # Negative because CMA-ES minimizes
            losses.append(loss)
            
            if solution_idx % 10 == 0:
                print(f"Evaluated solution {solution_idx + 1}/{len(sols)}, reward: {total_reward:.2f}")
            
            # Log to CSV: generation, individual, score
            csv_writer.writerow([generation, solution_idx + 1, float(total_reward)])
            csv_file.flush()
        
        return np.array(losses)

    # Main training loop
    generation = 0
    best_reward = float("-inf")
    
    # Initialize tracking
    history = {
        'generation': [],
        'best_reward': [],
        'mean_reward': [],
        'worst_reward': [],
        'std_reward': [],
        'best_ever': [],
        'sigma': [],
    }
    
    # Create output directory
    output_dir = os.path.join('outputs', cfg.game.split('/')[-1], time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)
    
    while not es.stop() or generation < cfg.min_gen:
        sols = es.ask()
        losses = evaluate_population(jnp.array(sols), rng)
        rewards = -losses  # Convert back to rewards for logging
        
        # Update CMA-ES
        es.tell(sols, jnp.asarray(losses).tolist())
        
        # Save checkpoint
        checkpoint_dir = os.path.dirname(cfg.train_cfg.checkpoint)
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(cfg.train_cfg.checkpoint, "wb") as f:
            pickle.dump(es, f)
            
        # Update statistics
        gen_best = np.max(rewards)
        gen_mean = np.mean(rewards)
        gen_worst = np.min(rewards)
        gen_std = np.std(rewards)
        best_reward = max(best_reward, gen_best)
        
        # Update history
        history['generation'].append(generation)
        history['best_reward'].append(gen_best)
        history['mean_reward'].append(gen_mean)
        history['worst_reward'].append(gen_worst)
        history['std_reward'].append(gen_std)
        history['best_ever'].append(best_reward)
        history['sigma'].append(es.sigma)
        
        # Log progress
        print(f"\nGeneration {generation}")
        print(f"  Best fitness: {gen_best:.2f}")
        print(f"  Mean fitness: {gen_mean:.2f} ± {gen_std:.2f}")
        print(f"  Worst fitness: {gen_worst:.2f}")
        print(f"  Best ever: {best_reward:.2f}")
        print(f"  Step size (σ): {es.sigma:.2e}")
        
        # Save current statistics
        stats = {
            'generation': generation,
            'best_reward': float(gen_best),
            'mean_reward': float(gen_mean),
            'worst_reward': float(gen_worst),
            'std_reward': float(gen_std),
            'best_ever': float(best_reward),
            'sigma': float(es.sigma),
            'stopped': es.stop(),
            'stop_dict': es.stop().__dict__ if es.stop() else None
        }
        
        with open(os.path.join(output_dir, 'latest_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)
            
        # Plot progress every 10 generations
        if generation % 10 == 0:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
            
            # Fitness plot
            ax1.fill_between(history['generation'], 
                           np.array(history['mean_reward']) - np.array(history['std_reward']),
                           np.array(history['mean_reward']) + np.array(history['std_reward']),
                           alpha=0.2)
            ax1.plot(history['generation'], history['best_reward'], label='Best')
            ax1.plot(history['generation'], history['mean_reward'], label='Mean')
            ax1.plot(history['generation'], history['worst_reward'], label='Worst')
            ax1.plot(history['generation'], history['best_ever'], label='Best Ever', linestyle='--')
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Reward')
            ax1.legend()
            ax1.grid(True)
            
            # Step size plot
            ax2.semilogy(history['generation'], history['sigma'])
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Step Size (σ)')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'training_progress.png'))
            plt.close()
            
        generation += 1
    
    # Save final history
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
        
    # Create final plots
    plt.figure(figsize=(12, 8))
    plt.plot(history['generation'], history['best_reward'], label='Best')
    plt.plot(history['generation'], history['mean_reward'], label='Mean')
    plt.fill_between(history['generation'],
                    np.array(history['mean_reward']) - np.array(history['std_reward']),
                    np.array(history['mean_reward']) + np.array(history['std_reward']),
                    alpha=0.2)
    plt.plot(history['generation'], history['best_ever'], label='Best Ever', linestyle='--')
    plt.xlabel('Generation')
    plt.ylabel('Reward')
    plt.title(f'Training Progress on {cfg.game}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'final_progress.png'))
    plt.close()
    
    env.close()
    try:
        csv_file.close()
    except Exception:
        pass

if __name__ == '__main__':
    train()