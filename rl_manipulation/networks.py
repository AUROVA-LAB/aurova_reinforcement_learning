import torch as th
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomFeatureExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space, features_dim=128):
        super().__init__(
            observation_space,
            features_dim=features_dim,
        )

        input_dim = observation_space.shape[0]

        self.network_rot = nn.Sequential(
            nn.Linear(input_dim / 2 , 16),
            nn.LayerNorm(16),
            nn.Tanh(),

            nn.Linear(16, 32),
            nn.LayerNorm(32),
            nn.Tanh(),

            nn.Linear(32, features_dim),
            nn.Tanh(),
        )

        self.network_pos = nn.Sequential(
            nn.Linear(input_dim / 2 , 16),
            nn.LayerNorm(16),
            nn.Tanh(),

            nn.Linear(16, 32),
            nn.LayerNorm(32),
            nn.Tanh(),

            nn.Linear(32, features_dim),
            nn.Tanh(),
        )

    def forward(self, observations):
        rot = observations[:, :3]
        pos = observations[:, 3:]

        rot = self.network_rot(rot)
        pos = self.network_rot(pos)
        
        return th.cat((rot, pos), dim=-1)