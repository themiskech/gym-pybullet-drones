import os
import cv2
import numpy as np
from gym_pybullet_drones.envs.VisionAviary import VisionAviary

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run_camera_test():
    env = VisionAviary(gui=True, record=False, obstacles=True)
    obs = env.reset()

    try:
        for step in range(500):
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            image = obs["image"]
            if image is not None:
                image = image.transpose(1, 2, 0)  # CHW to HWC
                image = image.astype(np.uint8)
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imshow("Drone FPV Camera", image_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if done:
                obs = env.reset()

    finally:
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_camera_test()