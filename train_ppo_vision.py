import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYBULLET_EGL"] = "1"  
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_pybullet_drones.envs.VisionAviary import VisionAviary

# --- Parameters ---
TOTAL_TIMESTEPS = 500_000
MODEL_DIR = "./ppo_drone_vision_model"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Env factory ---
def make_env():
    return VisionAviary(gui=False, obstacles=True)

# --- Vec env ---
env = DummyVecEnv([make_env])

# --- PPO Model ---
model = PPO(
    policy="MultiInputPolicy",
    env=env,
    verbose=1,
    tensorboard_log=None,
    learning_rate=5e-4,
    n_steps=512,
    batch_size=64,
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# --- Train ---
model.learn(total_timesteps=TOTAL_TIMESTEPS)

# --- Save model ---
model.save(os.path.join(MODEL_DIR, "final_model"))
env.close()
