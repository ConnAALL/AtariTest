"""
SCOPE policy using CompressionLab DCT, NonLinearLab sparsification, and ShapingLab mapping.
Modified version of SCOPE.py that uses the lab implementations.
"""

import sys
from pathlib import Path
import argparse
import json
import numpy as np
import cma
import gymnasium as gym
import ale_py
import jax.numpy as jnp

# Add lab paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR

sys.path.insert(0, str(REPO_ROOT / "CompressionLab"))
sys.path.insert(0, str(REPO_ROOT / "NonLinearLab"))
sys.path.insert(0, str(REPO_ROOT / "ShapingLab"))

from compressionlab.dct import DCT2DBatchCompressor
from compressionlab.dft import DFTBatchCompressor
from compressionlab.wavelet import Wavelet2DBatchCompressor, pad_to_levels, crop_to_hw
from compressionlab.convolution import Conv2DCompressor
from compressionlab.gaussian import GaussianSplattingBatchCompressor
from NonLinearLab.nonlinearlab.sparsification import Sparsifier
from NonLinearLab.nonlinearlab.quantization import Quantizer
from NonLinearLab.nonlinearlab.DropoutRegularization import Dropout
from ShapingLab.shapinglab.shaping import ShapingMapper

class SCOPE_env:
    def __init__(self, chromosome: list, output_size: int, comp_cfg: dict, nonlinear_cfg: dict):
        """Create a new SCOPE policy instance."""
        self.comp_cfg = comp_cfg
        self.nonlinear_cfg = nonlinear_cfg
        # Required shaping/input sizes
        self.k = int(self.comp_cfg.get("k", 0))
        if self.k <= 0:
            raise ValueError("comp_cfg.k must be provided and > 0 for dct/dft shaping.")
        self.output_size = output_size
        self.compressor = None
        self.wavelet_levels = None  # Store levels for padding
        # Nonlinear operator selection
        nl_type = (self.nonlinear_cfg.get("type") or "sparsifier").lower()
        if nl_type == "sparsifier":
            q = float(self.nonlinear_cfg.get("q", 90.0))
            self.nonlinear = Sparsifier(q=q)
        elif nl_type == "quantizer":
            levels = int(self.nonlinear_cfg.get("num_levels", 256))
            self.nonlinear = Quantizer(num_levels=levels)
        elif nl_type == "dropout":
            rate = float(self.nonlinear_cfg.get("rate", 0.2))
            # For deterministic eval, use a fixed key=None which no-ops if rate==0; else raise
            self.nonlinear = Dropout(rate=rate, key=None)
        else:
            raise ValueError(f"Unsupported nonlinear type: {nl_type}")
        self.mapper = None
        self._process_chromosome(chromosome)

    def _process_chromosome(self, chromosome: list):
        """Split chromosome into weight and bias tensors"""
        w1_len = self.k
        w2_len = self.k * self.output_size
        self.weights_1 = np.asarray(chromosome[:w1_len], dtype=np.float32).reshape(1, self.k)
        self.weights_2 = np.asarray(chromosome[w1_len : w1_len + w2_len], dtype=np.float32).reshape(self.k, self.output_size)
        self.bias = np.asarray(chromosome[w1_len + w2_len :], dtype=np.float32).reshape(1, self.output_size)
        # Initialize ShapingLab mapper with JAX arrays
        w1_jnp = jnp.asarray(self.weights_1, dtype=jnp.float32)                  # (O=1, M=k)
        w2_jnp = jnp.asarray(self.weights_2, dtype=jnp.float32).reshape(self.k, 1, self.output_size)  # (N=k, C=1, P=output)
        b_jnp = jnp.asarray(self.bias, dtype=jnp.float32)                        # (O=1, P=output)
        self.mapper = ShapingMapper(w1_jnp, w2_jnp, b_jnp)

    def forward(self, frame: np.ndarray) -> np.ndarray:
        """Forward pass for the SCOPE policy"""
        if frame.max() > 1.0:
            frame = frame.astype(np.float32) / 255.0
            
        # Lazy-build compressor with image dimensions
        if self.compressor is None:
            h, w = frame.shape
            comp_type = self.comp_cfg.get("type", "wavelet").lower()
            
            if comp_type == "dct":
                self.compressor = DCT2DBatchCompressor(
                    channel_last=bool(self.comp_cfg.get("channel_last", True)),
                    include_channels=False,
                    norm=self.comp_cfg.get("norm", "ortho"),
                    k=self.k,
                    image_height=h,
                    image_width=w,
                    image_channels=int(self.comp_cfg.get("channels", 1)),
                )
            elif comp_type == "dft":
                self.compressor = DFTBatchCompressor(
                    channel_last=bool(self.comp_cfg.get("channel_last", True)),
                    include_channels=False,
                    norm=self.comp_cfg.get("norm", "ortho"),
                    real_output=False,
                    k=self.k,
                )
            elif comp_type == "wavelet":
                self.wavelet_levels = int(self.comp_cfg.get("wavelet_levels", 3))
                self.compressor = Wavelet2DBatchCompressor(
                    wavelet=self.comp_cfg.get("wavelet", "bior4.4"),
                    levels=self.wavelet_levels,
                    mode=self.comp_cfg.get("mode", "reflect"),
                    channel_last=bool(self.comp_cfg.get("channel_last", True)),
                )
            else:
                raise ValueError(f"Unsupported compressor type: {comp_type}")
        
        frame_batch = frame[None, ...]
        
        # For wavelet, pad the input to satisfy dimension requirements
        if self.wavelet_levels is not None:
            frame_batch_jax = jnp.array(frame_batch)
            frame_padded, hw_orig = pad_to_levels(frame_batch_jax, self.wavelet_levels, channel_last=True)
            coeffs = self.compressor(frame_padded)
        else:
            coeffs = self.compressor(frame_batch)

        # Coeffs may have batch/channel dims; extract 2D k×k block
        if coeffs.ndim == 3:  # (B, k, k)
            coeffs_2d = coeffs[0]
        elif coeffs.ndim == 4:  # (B, k, k, C)
            coeffs_2d = coeffs[0, :, :, 0]
        else:
            coeffs_2d = coeffs.squeeze()
            if coeffs_2d.ndim != 2:
                raise ValueError(f"Unexpected compressor output shape: {coeffs.shape}")

        # If complex (DFT), convert to magnitude to get real-valued features
        if jnp.iscomplexobj(coeffs_2d):
            coeffs_2d = jnp.abs(coeffs_2d)

        # Ensure k×k top-left block for shaping mapper (DFT may retain full H×W)
        coeffs_2d = coeffs_2d[: self.k, : self.k]

        # Prepare input for ShapingMapper: (B=1, M=k, N=k, C=1)
        X = jnp.asarray(np.asarray(coeffs_2d)[None, ..., None], dtype=jnp.float32)
        # Apply nonlinear operator (JAX op) and shaping mapper
        X_sparse = self.nonlinear(X)
        Y = self.mapper(X_sparse)  # (1, O=1, P=output_size)
        logits = np.asarray(Y[0, 0])  # to numpy for downstream usage
        return logits.flatten()


def compute_chromosome_size(k: int, output_size: int) -> int:
    """Return expected length of chromosome for the current SCOPE policy"""
    return k + k * output_size + output_size

def make_env(game_name: str, repeat_action_prob: float, frameskip: int) -> gym.Env:
    """Create a Gym environment for the given game"""
    return gym.make(id=game_name, obs_type="grayscale", repeat_action_probability=repeat_action_prob, frameskip=frameskip)

def evaluate_individual(solution: list, game: str, output_size: int, config: dict, comp_cfg: dict, nonlinear_cfg: dict) -> float:
    """Evaluate an individual policy"""
    policy = SCOPE_env(chromosome=solution, output_size=output_size, comp_cfg=comp_cfg, nonlinear_cfg=nonlinear_cfg)
    env = make_env(game_name=game, repeat_action_prob=config["REPEAT_ACTION_PROBABILITY"], frameskip=config["FRAMESKIP"])
    
    total_reward = 0.0

    for _ in range(config["EPISODES_PER_INDIVIDUAL"]):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0

        while not done and steps < config["MAX_STEPS_PER_EPISODE"]:
            state = obs.astype(np.float32) / 255.0
            scope_output = policy.forward(state)
            action = int(np.argmax(scope_output))
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            steps += 1

        total_reward += ep_reward

    env.close()
    return total_reward / config["EPISODES_PER_INDIVIDUAL"]

def main():
    """Run the SCOPE policy with lab implementations"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "configs" / "001_dct-sparsifier-ShapingMapper.json"))
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    comp_cfg = cfg.get("comp_cfg", {})
    nonlinear_cfg = cfg.get("nonlinear_cfg", {"type": "sparsifier", "q": 90.0})
    # Training/eval config mapping
    CONFIG = {
        "ENV_NAME": cfg.get("game", "ALE/SpaceInvaders-v5"),
        "EPISODES_PER_INDIVIDUAL": 1,
        "MAX_STEPS_PER_EPISODE": int(cfg.get("steps", 10000)),
        "POPULATION_SIZE": cfg.get("popsize", cfg.get("population_size", None)),
        "CMA_SIGMA": 0.5,
        "GENERATIONS": int(cfg.get("min_gen", 50)),
        "REPEAT_ACTION_PROBABILITY": float(cfg.get("repeat_action_prob", 0.0)),
        "FRAMESKIP": int(cfg.get("frameskip", 4)),
    }
    
    game = CONFIG["ENV_NAME"]
    env = make_env(game_name=game, repeat_action_prob=CONFIG["REPEAT_ACTION_PROBABILITY"], frameskip=CONFIG["FRAMESKIP"])
    output_size = env.action_space.n
    k = int(comp_cfg.get("k", 0))
    if k <= 0:
        raise ValueError("comp_cfg.k must be provided and > 0 for dct/dft shaping.")
    chromosome_size = compute_chromosome_size(k=k, output_size=output_size)
    env.close()

    popsize = CONFIG["POPULATION_SIZE"]
    inopts = {}
    if popsize is not None:
        inopts["popsize"] = int(popsize)
    es = cma.CMAEvolutionStrategy(x0=np.zeros(chromosome_size), sigma0=CONFIG["CMA_SIGMA"], inopts=inopts)

    best_overall_reward = float("-inf")

    for generation in range(CONFIG["GENERATIONS"]):
        solutions = es.ask()
        rewards = []

        for index, solution in enumerate(solutions):
            avg_reward = evaluate_individual(solution, game, output_size, CONFIG, comp_cfg, nonlinear_cfg)
            rewards.append(-avg_reward)  # Negative fitness as CMA-ES minimizes

            if avg_reward > best_overall_reward:
                best_overall_reward = avg_reward

            print(f"[GEN {generation+1}] Indv {index+1}/{len(solutions)}: Reward = {avg_reward:.2f}")

        es.tell(solutions, rewards)

        best_gen_reward = -min(rewards)
        avg_gen_reward = -np.mean(rewards)
        print(f"[GEN {generation+1}] Best: {best_gen_reward:.2f} | Avg: {avg_gen_reward:.2f} | Best Overall: {best_overall_reward:.2f}")

if __name__ == "__main__":
    main()