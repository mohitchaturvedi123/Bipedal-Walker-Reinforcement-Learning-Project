from stable_baselines3 import SAC
import gymnasium as gym

env = gym.make("BipedalWalker-v3", render_mode="human")
model = SAC.load(r"D:\Personal Projects\Old Projects\Reinforcement Learning Projects\BIpedal Walker\best_model\best_model.zip")

obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
