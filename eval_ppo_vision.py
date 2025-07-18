import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.pop("PYBULLET_EGL", None)  # Επιτρέπει GUI

import cv2
import numpy as np
import torch
from stable_baselines3 import PPO
from gym_pybullet_drones.envs.VisionAviary import VisionAviary

# --- Load model ---
MODEL_PATH = "./ppo_drone_vision_model/final_model.zip"
model = PPO.load(MODEL_PATH, device="cuda" if torch.cuda.is_available() else "cpu")

# --- GUI eval ---
env = VisionAviary(gui=True, obstacles=True)
obs, info = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)

    # --- Print info ---
    pos = obs["features"][0:3]
    print(f"Drone Pos: {pos} | Reward: {reward:.2f}")

    # --- Show FPV ---
    img = obs["image"].transpose(1, 2, 0).astype(np.uint8)
    img = cv2.resize(img, (512, 512))
    cv2.imshow("Drone FPV", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    if done or truncated:
        print("[INFO] Episode done, resetting...")
        obs, info = env.reset()

env.close()
cv2.destroyAllWindows()
