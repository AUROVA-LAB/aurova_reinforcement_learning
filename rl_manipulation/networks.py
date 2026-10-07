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

        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LayerNorm(32),
            nn.Tanh(),

            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.Tanh(),

            nn.Linear(64, features_dim),
            nn.Tanh(),
        )

    def forward(self, observations):
        return self.network(observations)