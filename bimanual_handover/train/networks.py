import torch as th
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO

# Define a custom feature extractor (if needed)
class CustomFeatureExtractor(nn.Module):
    def __init__(self, observation_space):
        super(CustomFeatureExtractor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)

# Define a custom policy
class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            **kwargs,
        )
        # self.mlp_extractor = CustomFeatureExtractor(observation_space=th.zeros(4))

        features = 64
        action_shape = 2

        if net_arch is not None:
            features = net_arch[-1]

        if action_space.shape is not ():
            action_shape = action_space.shape[0]

        # Replace the actor with your custom network
        self.action_net = nn.Sequential(
            nn.Linear(features, action_shape),
            nn.Tanh())

        # Reinitialize parameters (important)
        self.action_net.apply(self.init_weights)

# Instantiate the PPO model with your custom policy
model = PPO(
    policy=CustomActorCriticPolicy,
    env="CartPole-v1",  # Replace with your environment
    verbose=1,
)
print(model.policy)

# Train the model
# model.learn(total_timesteps=10000)
