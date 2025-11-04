"""
SCOPE policy using CompressionLab DCT, NonLinearLab sparsification, and ShapingLab mapping.
Modified version of SCOPE.py that uses the lab implementations.
"""

import sys
from pathlib import Path
import numpy as np
import cma
import gymnasium as gym
import ale_py

# Add lab paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR

sys.path.insert(0, str(REPO_ROOT / "CompressionLab"))
sys.path.insert(0, str(REPO_ROOT / "NonLinearLab"))

from compressionlab.dct import DCT2DBatchCompressor
from nonlinearlab.sparsification import Sparsifier


class SCOPE_env:
    def __init__(self, chromosome: list, k: int, p: float, output_size: int):
        """Create a new SCOPE policy instance."""
        self.k = k
        self.p = p
        self.output_size = output_size
        self.dct = None
        self.sparsifier = Sparsifier(q=self.p)
        self._process_chromosome(chromosome)

    def _process_chromosome(self, chromosome: list):
        """Split chromosome into weight and bias tensors"""
        w1_len = self.k
        w2_len = self.k * self.output_size
        self.weights_1 = np.asarray(chromosome[:w1_len], dtype=np.float32).reshape(1, self.k)
        self.weights_2 = np.asarray(chromosome[w1_len : w1_len + w2_len], dtype=np.float32).reshape(self.k, self.output_size)
        self.bias = np.asarray(chromosome[w1_len + w2_len :], dtype=np.float32).reshape(1, self.output_size)

    def forward(self, frame: np.ndarray) -> np.ndarray:
        """Forward pass for the SCOPE policy"""
        if frame.max() > 1.0:
            frame = frame.astype(np.float32) / 255.0
            
        if self.dct is None:
            h, w = frame.shape
            self.dct = DCT2DBatchCompressor(channel_last=True, include_channels=False, norm="ortho", k=self.k, image_height=h, image_width=w, image_channels=1)
        
        frame_batch = frame[None, ...]
        coeffs = self.dct(frame_batch)
        
        if coeffs.shape != (self.k, self.k):
            coeffs = coeffs[:self.k, :self.k]
        
        coeffs_batch = coeffs[None, None, ...]
        coeffs_sparse = self.sparsifier(coeffs_batch)
        m_prime = np.asarray(coeffs_sparse[0, 0])
        
        logits = self.weights_1 @ m_prime @ self.weights_2 + self.bias
        return logits.flatten()


def compute_chromosome_size(k: int, output_size: int) -> int:
    """Return expected length of chromosome for the current SCOPE policy"""
    return k + k * output_size + output_size

def make_env(game_name: str, repeat_action_prob: float, frameskip: int) -> gym.Env:
    """Create a Gym environment for the given game"""
    return gym.make(id=game_name, obs_type="grayscale", repeat_action_probability=repeat_action_prob, frameskip=frameskip)

def evaluate_individual(solution: list, game: str, output_size: int, config: dict) -> float:
    """Evaluate an individual policy"""
    policy = SCOPE_env(chromosome=solution, k=config["K"], p=config["P"], output_size=output_size)
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
    CONFIG = {
        "ENV_NAME": "ALE/SpaceInvaders-v5",
        "EPISODES_PER_INDIVIDUAL": 1,
        "MAX_STEPS_PER_EPISODE": 10000,
        "POPULATION_SIZE": None,
        "CMA_SIGMA": 0.5,
        "GENERATIONS": 5000,
        "K": 75,
        "P": 25,
        "REPEAT_ACTION_PROBABILITY": 0.0,
        "FRAMESKIP": 4
    }
    
    game = CONFIG["ENV_NAME"]
    env = make_env(game_name=game, repeat_action_prob=CONFIG["REPEAT_ACTION_PROBABILITY"], frameskip=CONFIG["FRAMESKIP"])
    output_size = env.action_space.n
    chromosome_size = compute_chromosome_size(k=CONFIG["K"], output_size=output_size)
    env.close()

    es = cma.CMAEvolutionStrategy(x0=np.zeros(chromosome_size), sigma0=CONFIG["CMA_SIGMA"], inopts={"popsize": CONFIG["POPULATION_SIZE"]})

    best_overall_reward = float("-inf")

    for generation in range(CONFIG["GENERATIONS"]):
        solutions = es.ask()
        rewards = []

        for index, solution in enumerate(solutions):
            avg_reward = evaluate_individual(solution, game, output_size, CONFIG)
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
