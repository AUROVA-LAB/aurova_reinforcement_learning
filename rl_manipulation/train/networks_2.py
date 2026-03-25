from torch import nn
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor


# 🔧 Custom MLP Extractor with:
# - LayerNorm
# - Split policy heads
class CustomMlpExtractor_2(nn.Module):
    def __init__(self, feature_dim, net_arch, activation_fn, device):
        super().__init__()

        self.device = device
        self.activation_fn = activation_fn

        last_dim = net_arch[-1]  # e.g. 256
        half_dim = last_dim // 2

        # ===== POLICY HEADS (split input) =====

        def build_head(input_dim):
            layers = []
            prev_dim = input_dim

            for size in net_arch:
                layers.append(nn.Linear(prev_dim, size))
                layers.append(nn.LayerNorm(size))
                layers.append(activation_fn())
                prev_dim = size

            return nn.Sequential(*layers)

        # Each head gets HALF of the features
        self.policy_head_1 = build_head(feature_dim // 2)
        self.policy_head_2 = build_head(feature_dim // 2)

        # ===== VALUE NETWORK (standard, full input) =====
        value_layers = []
        prev_dim = feature_dim

        for size in net_arch:
            value_layers.append(nn.Linear(prev_dim, size))
            value_layers.append(nn.LayerNorm(size))
            value_layers.append(activation_fn())
            prev_dim = size

        self.value_net = nn.Sequential(*value_layers)

        # Output dims expected by SB3
        self.latent_dim_pi = last_dim * 2  # concatenated heads
        self.latent_dim_vf = last_dim

    def forward(self, features):
        # Split input
        half = features.shape[1] // 2
        f1 = features[:, :half]
        f2 = features[:, half:]

        # Pass through heads
        p1 = self.policy_head_1(f1)
        p2 = self.policy_head_2(f2)

        # Concatenate outputs
        latent_pi = torch.cat([p1, p2], dim=1)

        # Value path (no split)
        latent_vf = self.value_net(features)

        return latent_pi, latent_vf

    def forward_actor(self, features):
        half = features.shape[1] // 2
        f1 = features[:, :half]
        f2 = features[:, half:]

        p1 = self.policy_head_1(f1)
        p2 = self.policy_head_2(f2)

        return torch.cat([p1, p2], dim=1)

    def forward_critic(self, features):
        return self.value_net(features)


# 🧠 Custom Policy
class CustomActorCriticPolicy_2(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            **kwargs,
        )

        # Determine feature size
        features = 64
        if net_arch is not None:
            features = net_arch[-1]

        # Action dimension
        if action_space.shape == ():
            action_dim = 1
        else:
            action_dim = action_space.shape[0]

        # ⚠️ latent_dim_pi is now doubled (because of concat)
        latent_pi_dim = features * 2

        # 🎯 Action head
        self.action_net = nn.Sequential(
            nn.Linear(latent_pi_dim, action_dim),
            nn.LayerNorm(action_dim),
            nn.Tanh()
        )

        # 🎯 Value head
        self.value_net = nn.Sequential(
            nn.Linear(features, 1),
            nn.LayerNorm(1)
        )

        self.action_net.apply(self.init_weights)
        self.value_net.apply(self.init_weights)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomMlpExtractor_2(
            self.features_dim,
            net_arch=[32, 64, 128],
            activation_fn=nn.Tanh,
            device=self.device,
        )