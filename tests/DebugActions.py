import time
import numpy as np
import cv2

from gym_pybullet_drones.envs.VisionAviary import VisionAviary

# --- Περιβάλλον ---
env = VisionAviary(gui=True, obstacles=True)
obs = env.reset()

# --- Πόσα frames να τρέξει ---
EPISODE_STEPS = 300
DELAY = 1 / 60

for i in range(EPISODE_STEPS):
    action = env.action_space.sample()  # <--- Τυχαία ενέργεια
    print(f"[STEP {i}] Action sent to drone:", action)

    obs, reward, done, truncated, info = env.step(action)

    drone_pos = env._getDroneStateVector(0)[0:3]
    dist = np.linalg.norm(drone_pos - env.goal)
    print(f"[STEP {i}] Drone position: {drone_pos}, distance to goal: {dist:.2f}, reward: {reward:.2f}")

    # --- Εικόνα από κάμερα ---
    try:
        img = obs.transpose(1, 2, 0)  # CHW -> HWC
        img = cv2.resize(img, (512, 512))
        cv2.imshow("Drone Camera", img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    except Exception as e:
        print("[ERROR] Image processing failed:", str(e))

    if done or truncated:
        print("[INFO] Episode terminated or truncated")
        break

    time.sleep(DELAY)

env.close()
cv2.destroyAllWindows()
