import os
import re

import gymnasium as gym
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import SAC

# ------------------------------------------------------------------ #
# Paths
# ------------------------------------------------------------------ #
BASE_DIR = r"D:\Personal Projects\Old Projects\Reinforcement Learning Projects\Bipedal Walker Reinforcement Learning"
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
BEST_MODEL_PATH = os.path.join(BASE_DIR, "best_model", "best_model.zip")
FINAL_MODEL_PATH = os.path.join(BASE_DIR, "sac_bipedalwalker_final.zip")
GIF_DIR = os.path.join(BASE_DIR, "CheckpointsGif")
os.makedirs(GIF_DIR, exist_ok=True)

# ------------------------------------------------------------------ #
# Settings (tweak these to control quality / file size)
# LinkedIn GIFs should ideally stay under ~5 MB each.
# ------------------------------------------------------------------ #
SEED = 42            # same terrain for every model -> fair comparison
MAX_STEPS = 800      # cap episode length so GIFs stay small
FRAME_SKIP = 2       # keep every Nth frame
OUT_WIDTH = 480      # GIF width in pixels (height keeps aspect ratio)
GIF_FPS = 25
COLORS = 128         # palette size (lower = smaller file)
MAKE_COMBINED = True # also create one GIF with all checkpoints in order
COMBINED_FRAME_SKIP = 2  # extra subsampling for the combined GIF


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def get_font(size):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT = get_font(18)


def step_number(path):
    match = re.search(r"_(\d+)_steps", os.path.basename(path))
    return int(match.group(1)) if match else -1


def label_for(path):
    name = os.path.basename(path).lower()
    if "best_model" in name:
        return "Best Model"
    if "final" in name:
        return "Final Model"
    return f"{step_number(path):,} steps"


def annotate(frame, title, reward):
    """Resize the frame and draw a label banner on top."""
    img = Image.fromarray(frame)
    height = int(img.height * OUT_WIDTH / img.width)
    img = img.resize((OUT_WIDTH, height), Image.LANCZOS)

    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, OUT_WIDTH, 34], fill=(20, 20, 30))
    draw.text((10, 7), f"SAC | {title}", font=FONT, fill=(255, 255, 255))
    reward_text = f"Reward: {reward:.0f}"
    text_w = draw.textlength(reward_text, font=FONT)
    color = (80, 220, 120) if reward > 0 else (255, 110, 110)
    draw.text((OUT_WIDTH - text_w - 10, 7), reward_text, font=FONT, fill=color)
    return img


def save_gif(frames, path):
    palette_frames = [
        f.convert("P", palette=Image.ADAPTIVE, colors=COLORS) for f in frames
    ]
    palette_frames[0].save(
        path,
        save_all=True,
        append_images=palette_frames[1:],
        duration=int(1000 / GIF_FPS),
        loop=0,
        optimize=True,
    )
    size_mb = os.path.getsize(path) / (1024 * 1024)
    warn = "  <-- over 5 MB, lower OUT_WIDTH/MAX_STEPS" if size_mb > 5 else ""
    print(f"   saved {os.path.basename(path)} ({size_mb:.2f} MB){warn}")


def record_episode(model, env, title):
    obs, _ = env.reset(seed=SEED)
    frames, total_reward = [], 0.0

    for step in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if step % FRAME_SKIP == 0:
            frames.append(annotate(env.render(), title, total_reward))

        if terminated or truncated:
            break

    # Hold the last frame briefly so viewers can see the outcome
    frames.extend([frames[-1]] * (GIF_FPS // 2))
    return frames, total_reward


# ------------------------------------------------------------------ #
# Collect models
# ------------------------------------------------------------------ #
checkpoint_files = sorted(
    [
        os.path.join(CHECKPOINT_DIR, f)
        for f in os.listdir(CHECKPOINT_DIR)
        if f.endswith(".zip")
    ],
    key=step_number,
)

model_files = checkpoint_files + [FINAL_MODEL_PATH, BEST_MODEL_PATH]

# ------------------------------------------------------------------ #
# Run and record
# ------------------------------------------------------------------ #
env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
combined_frames = []

for model_path in model_files:
    title = label_for(model_path)
    print(f"\n=== {title}: {os.path.basename(model_path)} ===")

    model = SAC.load(model_path)
    frames, total_reward = record_episode(model, env, title)
    print(f"   reward: {total_reward:.1f} | frames: {len(frames)}")

    safe_name = os.path.splitext(os.path.basename(model_path))[0]
    save_gif(frames, os.path.join(GIF_DIR, f"{safe_name}.gif"))

    if MAKE_COMBINED:
        combined_frames.extend(frames[::COMBINED_FRAME_SKIP])

env.close()

if MAKE_COMBINED and combined_frames:
    print("\n=== Combined evolution GIF ===")
    save_gif(combined_frames, os.path.join(GIF_DIR, "00_training_evolution.gif"))

print(f"\nDone! GIFs saved in: {GIF_DIR}")
