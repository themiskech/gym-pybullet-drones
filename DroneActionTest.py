import os
import time
import numpy as np
import cv2

from gym_pybullet_drones.envs.VisionAviary import VisionAviary

# --- Parameters ---
DELAY = 1 / 48

# --- Setup ---
env = VisionAviary(gui=True, obstacles=True)
obs = env.reset()

# --- Test Actions ---
# Each action has 4 values: [thrust0, thrust1, thrust2, thrust3]
actions_to_test = [
    np.array([[0.0, 0.0, 0.0, 0.0]]),   # Do nothing
    np.array([[0.5, 0.5, 0.5, 0.5]]),   # Ascend
    np.array([[1.0, 1.0, 1.0, 1.0]]),   # Max thrust
    np.array([[0.8, 0.6, 0.8, 0.6]]),   # Tilt forward
    np.array([[0.6, 0.8, 0.6, 0.8]]),   # Tilt backward
    np.array([[0.6, 0.6, 0.8, 0.8]]),   # Roll right
    np.array([[0.8, 0.8, 0.6, 0.6]])    # Roll left
]

for idx, action in enumerate(actions_to_test):
    print(f"[INFO] Testing action {idx+1}: {action}")
    obs = env.reset()
    for _ in range(100):  # Step for a short duration
        obs, _, done, _, _ = env.step(action)

        try:
            if obs.shape[0] == 3:
                img = obs.transpose(1, 2, 0)
            else:
                img = obs  # ήδη HWC

            img = cv2.resize(img, (512, 512))
            cv2.imshow("Drone Camera", img)
        except Exception as e:
            print("[ERROR] Cannot process observation image:", e)


        if cv2.waitKey(1) & 0xFF == 27:
            break
        time.sleep(DELAY)

env.close()
cv2.destroyAllWindows()