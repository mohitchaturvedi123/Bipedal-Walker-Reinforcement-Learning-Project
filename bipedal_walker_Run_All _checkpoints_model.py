import os
from stable_baselines3 import SAC
import gymnasium as gym

# Base paths
BASE_DIR = r"D:\Personal Projects\Old Projects\Reinforcement Learning Projects\BIpedal Walker"
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
BEST_MODEL_PATH = os.path.join(BASE_DIR, "best_model", "best_model.zip")
FINAL_MODEL_PATH = os.path.join(BASE_DIR, "sac_bipedalwalker_final.zip")

# Collect all checkpoint files (sorted by step number)
checkpoint_files = sorted(
    [os.path.join(CHECKPOINT_DIR, f) for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".zip")],
    key=lambda x: int(x.split("_")[-2])  # assumes filenames like sac_bipedalwalker_200000_steps.zip
)

# Add final + best model at the end
model_files = checkpoint_files + [FINAL_MODEL_PATH, BEST_MODEL_PATH]

# Create environment with rendering
env = gym.make("BipedalWalker-v3", render_mode="human")

# Run each model sequentially
for model_path in model_files:
    print(f"\n=== Running model: {model_path} ===")
    model = SAC.load(model_path)

    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

env.close()
