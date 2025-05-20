import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import numpy as np
import cv2
import imageio
import pybullet as p

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from gym_pybullet_drones.envs.VisionAviary import VisionAviary

# --- Parameters ---
MODEL_PATH = "ppo_drone_vision.zip"
NUM_EPISODES = 5
RENDER = True           # ✅ GUI ενεργό
SHOW_CAM = True
SAVE_VIDEO = True
DELAY = 1 / 48

# --- Environment wrapper ---
def make_env():
    try:
        return VisionAviary(gui=RENDER, obstacles=True)
    except p.error as e:
        print("[WARNING] PyBullet GUI failed. Switching to DIRECT mode.")
        return VisionAviary(gui=False, obstacles=True)

env = DummyVecEnv([make_env])
#env = VecTransposeImage(env)

# --- Load trained model ---
model = PPO.load(MODEL_PATH)

# --- Evaluation loop ---
for ep in range(NUM_EPISODES):
    obs = env.reset()
    done = False
    total_reward = 0
    frames = []

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]

        img = obs[0].transpose(1, 2, 0)
        img = cv2.resize(img, (1280, 720))

        if SHOW_CAM:
            cv2.imshow("Drone Camera", img)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        if SAVE_VIDEO:
            frames.append(img)

        time.sleep(DELAY)

    print(f"[INFO] Episode {ep+1} total reward: {total_reward:.2f}")

    if SAVE_VIDEO and len(frames) > 0:
        video_path = f"eval_episode_{ep+1}.mp4"
        imageio.mimsave(video_path, frames, fps=24)
        print(f"[INFO] Saved video to {video_path}")

env.close()
cv2.destroyAllWindows()