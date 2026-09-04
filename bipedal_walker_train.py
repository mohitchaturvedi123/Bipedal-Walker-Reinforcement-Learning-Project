import os
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = r"D:\Personal Projects\Old Projects\Reinforcement Learning Projects\BIpedal Walker"
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
BEST_MODEL_DIR = os.path.join(BASE_DIR, "best_model")
LOG_DIR = os.path.join(BASE_DIR, "logs")
FINAL_MODEL_PATH = os.path.join(BASE_DIR, "sac_bipedalwalker_final")

for d in (BASE_DIR, CHECKPOINT_DIR, BEST_MODEL_DIR, LOG_DIR):
    os.makedirs(d, exist_ok=True)

ENV_ID = "BipedalWalker-v3"
N_ENVS = 8
TOTAL_TIMESTEPS = 2_500_000


def main():
    # ------------------------------------------------------------------
    # Training environment: parallel, no rendering (fast)
    # ------------------------------------------------------------------
    train_env = make_vec_env(ENV_ID, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)

    # ------------------------------------------------------------------
    # Separate evaluation environment (no rendering, just used to
    # periodically score the current policy and save the best model)
    # ------------------------------------------------------------------
    eval_env = gym.make(ENV_ID)

    model = SAC(
        "MlpPolicy",
        train_env,
        verbose=1,
        batch_size=256,
        buffer_size=1_00_000,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,+
        train_freq=1,Reinforcement Learning Projects/BIpedal Walker/bipedal_walker_ver2_5afast_RUNSAVEDMODEL.py
        gradient_steps=1,
        learning_starts=10_000,
        tensorboard_log=LOG_DIR,
    )

    # Saves a checkpoint every `save_freq` env steps (per-env steps, so
    # divide by n_envs to get an interval in terms of total timesteps)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(200_000 // N_ENVS, 1),
        save_path=CHECKPOINT_DIR,
        name_prefix="sac_bipedalwalker",
    )

    # Runs 20 evaluation episodes every 100k total timesteps, saves the
    # best-performing model separately, no rendering (headless, fast)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=max(100_000 // N_ENVS, 1),
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )

    callback = CallbackList([checkpoint_callback, eval_callback])

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

    model.save(FINAL_MODEL_PATH)
    print(f"Final model saved to: {FINAL_MODEL_PATH}.zip")
    print(f"Best model saved to: {os.path.join(BEST_MODEL_DIR, 'best_model.zip')}")
    print(f"Checkpoints saved to: {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
