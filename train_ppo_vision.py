import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from gym_pybullet_drones.envs.VisionAviary import VisionAviary

# --- Paths ---
LOG_DIR = "./ppo_drone_tensorboard/"
MODEL_PATH = "./ppo_drone_vision"
EVAL_FREQ = 5000
SAVE_FREQ = 10000
TOTAL_TIMESTEPS = 1_000_000

# --- Training Environment ---
def make_train_env():
    env = VisionAviary(gui=True, obstacles=True)
    env = Monitor(env)
    return env

env = DummyVecEnv([make_train_env])

# --- Evaluation Environment ---
def make_eval_env():
    eval_env = VisionAviary(gui=False, obstacles=True)
    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])
    return eval_env

eval_env = make_eval_env()

# --- Custom Callback for showing drone camera ---
class ShowDroneCameraCallback(BaseCallback):
    def __init__(self, env, interval=500, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.interval = interval
        self.counter = 0

    def _on_step(self) -> bool:
        self.counter += 1
        if self.counter % self.interval == 0:
            try:
                img = self.env.envs[0].env.getDroneImage()

                if img is not None:
                    if img.dtype != 'uint8':
                        img = (img * 255).clip(0, 255).astype('uint8')
                    cv2.imshow("Drone Camera", img)
                    cv2.waitKey(1)
            except Exception as e:
                print(f"[Camera Callback Error] {e}")
        return True

# --- Callbacks ---
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=MODEL_PATH + "/best_model",
    log_path=LOG_DIR,
    eval_freq=EVAL_FREQ,
    deterministic=True,
    render=False
)

checkpoint_callback = CheckpointCallback(
    save_freq=SAVE_FREQ,
    save_path=MODEL_PATH + "/checkpoints",
    name_prefix="ppo_drone"
)

camera_callback = ShowDroneCameraCallback(env, interval=500)

# --- Model Definition ---
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=2.5e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=4,
    clip_range=0.2,
    gamma=0.99,
    gae_lambda=0.95,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# --- Training ---
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[eval_callback, checkpoint_callback, camera_callback]
)

model.save(MODEL_PATH + "/final_model")
env.close()
eval_env.close()
