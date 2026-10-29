import torch as th
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor


class TwoBranchExtractor(nn.Module):

    def __init__(
        self,
        feature_dim: int,
        branch_dim: int = 64,
        critic_dim: int = 128,
    ):
        super().__init__()

        self.branch1_dim = feature_dim // 2
        self.branch2_dim = feature_dim - self.branch1_dim

        self.latent_dim_pi = 2 * branch_dim
        self.latent_dim_vf = critic_dim

        self.actor_branch1 = nn.Sequential(
            nn.Linear(self.branch1_dim, branch_dim),
            nn.Tanh(),
            nn.Linear(branch_dim, branch_dim),
            nn.Tanh(),
        )

        self.actor_branch2 = nn.Sequential(
            nn.Linear(self.branch2_dim, branch_dim),
            nn.Tanh(),
            nn.Linear(branch_dim, branch_dim),
            nn.Tanh(),
        )

        self.critic_net = nn.Sequential(
            nn.Linear(feature_dim, critic_dim),
            nn.Tanh(),
            nn.Linear(critic_dim, critic_dim),
            nn.Tanh(),
        )

    def forward_actor(self, features):
        x1 = features[:, :self.branch1_dim]
        x2 = features[:, self.branch1_dim:]

        z1 = self.actor_branch1(x1)
        z2 = self.actor_branch2(x2)

        # print(th.cat((z1, z2), dim=1).shape)
        # raise

        return th.cat((z1, z2), dim=1)

    def forward_critic(self, features):
        return self.critic_net(features)

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, **kwargs):

        # Initialize the standard SB3 policy.
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            **kwargs,
        )

        if len(observation_space.shape) != 1:
            raise ValueError(
                "Expected a flat observation vector (shape=(N,))."
            )

        # Replace the MLP extractor while preserving SB3's action
        # distribution and policy/value heads.
        self.mlp_extractor = TwoBranchExtractor(
            feature_dim=observation_space.shape[0],
            branch_dim=128,
            critic_dim=256,
        )

        # Determine the number of input features
        features = 64
        action_shape = 2

        if net_arch != None:
            features = net_arch[-1]

        if action_space.shape != ():
            action_shape = action_space.shape[0]

        # Replace the actor with your custom network
        self.action_net = nn.Sequential(
            nn.Linear(features, action_shape),
            nn.Tanh())

        self.value_net = nn.Sequential(
                    nn.Linear(features, 1),
                    nn.Tanh())
                    


