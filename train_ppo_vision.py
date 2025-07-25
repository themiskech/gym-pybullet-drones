import os
import torch
import gymnasium as gym
import numpy as np
from sb3_contrib.ppo_recurrent import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeDict
from gym_pybullet_drones.envs.VisionAviary import VisionAviary

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space["image"].shape[0]
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.Flatten()
        )
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space["image"].shape)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(n_flatten + observation_space["goal"].shape[0], features_dim),
            torch.nn.ReLU()
        )

    def forward(self, observations):
        cnn_out = self.cnn(observations["image"])
        goal = observations["goal"]
        return self.linear(torch.cat([cnn_out, goal], dim=1))

def make_env():
    def _init():
        return VisionAviary(gui=False, record=False, obstacles=True)
    return _init

if __name__ == "__main__":
    env = SubprocVecEnv([make_env()])
    env = VecTransposeDict(env)

    model = RecurrentPPO(
        policy=RecurrentActorCriticPolicy,
        env=env,
        verbose=1,
        tensorboard_log="./ppo_drone_tensorboard/",
        learning_rate=0.0003,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        device="cuda" if torch.cuda.is_available() else "cpu",
        policy_kwargs=dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[128, 64], vf=[128, 64])
        )
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./ppo_checkpoints/",
        name_prefix="drone_model"
    )

    model.learn(total_timesteps=500_000, callback=checkpoint_callback)
    model.save("ppo_drone_final")

