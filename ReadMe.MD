# 🏃‍♂️ Bipedal Walker SAC Training & Evaluation

This repository contains scripts to train and evaluate a **Soft Actor-Critic (SAC)** agent on the `BipedalWalker-v3` environment using [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).

---

## 📂 Project Structure

- **bipedal_walker_train.py**  
  Trains a SAC agent with parallel environments, checkpoint saving, evaluation, and TensorBoard logging.

- **bipedal_walker_Run_All_checkpoints_model.py**  
  Loads **all saved checkpoints**, the final model, and the best model sequentially, running each visually in the environment.

- **bipedal_walker_Run_Best_model.py**  
  Loads and runs the **best-performing model** with rendering enabled.

---

## ⚙️ Training Setup

- **Algorithm:** Soft Actor-Critic (SAC)  
- **Environment:** `BipedalWalker-v3`  
- **Parallel Envs:** 8 (`SubprocVecEnv`)  
- **Timesteps:** 2,500,000  
- **Batch Size:** 256  
- **Replay Buffer:** 100,000  
- **Learning Rate:** 3e-4  
- **Gamma:** 0.99  
- **Tau:** 0.005  
- **Evaluation:** 20 episodes every ~12,500 steps  
- **Checkpoints:** Saved every ~25,000 steps  
- **Logging:** TensorBoard (`logs/`)

---

## 📊 Training Results

Final evaluation after **2.5M timesteps**:

- **Mean reward:** 327.10 ± 0.67  
- **Mean episode length:** ~683 steps (near max of 700)  
- **Actor loss:** -36.7  
- **Critic loss:** 0.0917  
- **Entropy coefficient:** 0.0113  
- **Updates performed:** 311,249  

✅ The agent walks almost perfectly — surviving nearly the full episode length with strong reward.

---

## 📂 Saved Models

- **Final model:** `sac_bipedalwalker_final.zip`  
- **Best model:** `best_model.zip`  
- **Checkpoints:** multiple snapshots in `checkpoints/`

---

## ▶️ Usage

### 1. Train the agent
```bash
python bipedal_walker_train.py
