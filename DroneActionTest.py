import os
import time
import numpy as np
import cv2
from gym_pybullet_drones.envs.VisionAviary import VisionAviary

# --- Parameters ---
DELAY = 1 / 48

def build_action(direction_index, speed, duration_scalar):
    action = np.zeros(8, dtype=np.float32)
    action[direction_index] = 1.0  # one-hot direction
    action[6] = speed              # speed (0.0–1.0)
    action[7] = duration_scalar    # duration scalar (0.0–1.0)
    return action

# --- Test Actions (Box(8,)) ---
actions_to_test = [
    build_action(0, 0.8, 0.5),  # forward
    build_action(1, 0.8, 0.5),  # backward
    build_action(2, 0.8, 0.5),  # right
    build_action(3, 0.8, 0.5),  # left
    build_action(4, 1.0, 0.3),  # up
    build_action(5, 1.0, 0.3),  # down
    build_action(0, 0.0, 0.5),  # no thrust
]

# --- Environment Setup ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
env = VisionAviary(gui=True, obstacles=True)
obs = env.reset()

# --- Action Loop ---
for idx, action in enumerate(actions_to_test):
    print(f"[INFO] Testing action {idx+1}: {action.flatten()}")
    obs = env.reset()
    done = False
    step = 0

    while not done and step < 150:
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

        try:
            if isinstance(obs, dict):
                image = obs["image"].transpose(1, 2, 0).astype(np.uint8)
            else:
                image = obs.transpose(1, 2, 0).astype(np.uint8)
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("Drone FPV Camera", image_bgr)
        except Exception as e:
            print("[ERROR] Could not process image:", e)

        if cv2.waitKey(1) & 0xFF == 27:
            break
        time.sleep(DELAY)

# --- Cleanup ---
env.close()
cv2.destroyAllWindows()
