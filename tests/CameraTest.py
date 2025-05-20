import os
import cv2
import time
import numpy as np
from gym_pybullet_drones.envs.VisionAviary import VisionAviary

# --- Parameters ---
DELAY = 1 / 48

# --- Environment setup ---
env = VisionAviary(gui=True, obstacles=True)
obs = env.reset()

# --- Dummy action ---
action = np.array([[0.5, 0.5, 0.5, 0.5]])

while True:
    obs, reward, done, truncated, info = env.step(action)

    try:
        img = None
        if isinstance(obs, dict) and "rgb" in obs:
            img = obs["rgb"]
        elif isinstance(obs, np.ndarray):
            img = obs

        if img is not None and img.ndim == 3:
            if img.shape[0] == 3:
                img = img.transpose(1, 2, 0)
            img = img.astype(np.uint8)  # ensure uint8 type
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
            cv2.imshow("Drone Camera", img)
        else:
            print("[WARN] Invalid observation image:", None if img is None else img.shape)

    except Exception as e:
        print("[ERROR] Cannot process observation image:", str(e))

    if cv2.waitKey(1) & 0xFF == 27:
        break

    time.sleep(DELAY)

env.close()
cv2.destroyAllWindows()
