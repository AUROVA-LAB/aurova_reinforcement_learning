import torch as th
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import MlpExtractor
import os
import gymnasium as gym
import copy

from stable_baselines3.common.type_aliases import PyTorchObs
from typing import Any, Optional, TypeVar, Union

# Custom MlpFeatureExtractor
class CustomMlpExtractor(MlpExtractor):
    def __init__(self, feature_dim, net_arch, activation_fn, device, phase, option, prev_feature_extractor = nn.Sequential()):
        super().__init__(feature_dim, net_arch, activation_fn, device)

        # Filter by phases and options
        if True: #phase == 1:
            
            # Uses Option 1 arquitecture
            if option == 1:

                # Feature extractor from Approach phase
                self.prev_policy_net = prev_feature_extractor["policy_net"].to(device)
                self.prev_value_net = prev_feature_extractor["value_net"].to(device)
                
                # Builds combined feature extractor
                last_layer = self.prev_policy_net[-1].out_features + 16
                combined_extractor = []

                for curr_layer in net_arch:
                    combined_extractor.append(nn.Sequential(nn.Linear(last_layer, curr_layer)))
                    combined_extractor.append(activation_fn())
                    last_layer = curr_layer

                # Declares policy and value net
                self.policy_net = nn.Sequential(*combined_extractor).to(device)
                self.value_net = nn.Sequential(*copy.deepcopy(combined_extractor)).to(device)

                self.forward_actor = self.forward_actor_1
                self.forward_critic = self.forward_critic_1
     
     # Forward function for the actor
    def forward_actor_1(self, features: th.Tensor) -> th.Tensor:
        '''
        In:
            - features - torch.Tensor(batch, 7+7 ó 7+7+16): observations from the environment.

        Out:
            - latent - torch.Tensor(batch, net_arch[-1]): latent features processed from the observations. 
        '''
        
        # Process end effector and object position with previous feature extractor
        ee_obj = self.prev_policy_net(features[:, :14])

        # Return latent activations
        return self.policy_net(th.cat((ee_obj, features[:, 14:]), dim = -1))
    
    # Forward function for the critic
    def forward_critic_1(self, features: th.Tensor) -> th.Tensor:
        '''
        In:
            - features - torch.Tensor(batch, 7+7 ó 7+7+16): observations from the environment.

        Out:
            - latent - torch.Tensor(batch, net_arch[-1]): latent features processed from the observations. 
        '''
        
        # Process end effector and object position with previous feature extractor
        ee_obj = self.prev_value_net(features[:, :14])

        # Return latent activations
        return self.value_net(th.cat((ee_obj, features[:, 14:]), dim = -1))



def insert_bn_dropout(seq):
    layers = []
    for i, layer in enumerate(seq):
        layers.append(layer)
        if isinstance(layer, nn.Linear):
            layers.append(nn.BatchNorm1d(layer.out_features))
        # if isinstance(layer, nn.Tanh):
        #     layers.append(nn.Dropout(p=0.0))  # Dropout after Tanh
    return nn.Sequential(*layers)

# Define a custom policy
class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, **kwargs):
        
        # Extract the arguments for the phase filtering
        my_kwargs = kwargs.pop("my_kwargs")

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            **kwargs,
        )

        self.seq_len = my_kwargs["seq_len"]

        # If the phase is MANIPULATION
        # if True: # my_kwargs["phase"] == my_kwargs["MANIPULATION"]:
        #     if my_kwargs["option"] == 1:
                
        #         # Load pre-trained weights
        #         pretrained_path = os.path.join("/workspace/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/classic/aurova_reinforcement_learning/bimanual_handover/train/logs/", my_kwargs["path"], "policy.pth")
        #         weights = th.load(pretrained_path, map_location="cpu", weights_only = "True")

        #         net = {}
        #         net["policy_net"] = []
        #         net["value_net"] = []

        #         for i in weights.keys():
        #             if "mlp_extractor" in i and "bias" not in i:

        #                 layer = nn.Linear(weights[i].shape[-1], weights[i].shape[0])

        #                 layer_w = weights[i]
        #                 layer_bias_w = weights[".".join(i.split(".")[0:3]) + ".bias"]
                        
        #                 layer.weight.data = layer_w
        #                 if layer_bias_w is not None:
        #                     layer.bias.data = layer_bias_w

        #                 for param in layer.parameters():
        #                     param.requires_grad = False

        #                 net[i.split(".")[1]].append(layer)

        #         for i in net.keys():
        #             net[i] = nn.Sequential(*net[i])

        #         # Obtain the pre-trained feature extractor
        #         prev_feature_extractor = net
                
        #         # Creates the new custom feature extractor
        #         self.mlp_extractor = CustomMlpExtractor(feature_dim = self.mlp_extractor.policy_net[0].in_features, net_arch = net_arch, activation_fn = kwargs["activation_fn"], device = self.device, prev_feature_extractor = prev_feature_extractor, phase = my_kwargs["phase"], option = my_kwargs["option"])
        
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

        # Reinitialize parameters (important)
        self.action_net.apply(self.init_weights)

        self.mlp_extractor.policy_net = nn.LSTM(input_size = 6, hidden_size = 64, batch_first = True)
        
        self.mlp_extractor.value_net = nn.LSTM(input_size = 6, hidden_size = 64, batch_first = True)
        


        # self.mlp_extractor.policy_net = insert_bn_dropout(self.mlp_extractor.policy_net)
        # self.mlp_extractor.value_net = insert_bn_dropout(self.mlp_extractor.value_net)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        obs = obs.view(obs.shape[0], self.seq_len, -1)
        
        if self.share_features_extractor:

            latent_pi = self.mlp_extractor.value_net(obs[:, :-1])[1][0].squeeze(0)
            latent_vf = self.mlp_extractor.policy_net(obs[:, :-1])[1][0].squeeze(0)

        else:
            pi_features, vf_features = obs
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations        
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]

        return actions, values, log_prob
    

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        obs = obs.view(obs.shape[0], self.seq_len, -1)

        latent_vf = self.mlp_extractor.forward_critic(obs)[1][0].squeeze(0)

        return self.value_net(latent_vf)


    def evaluate_actions(self, obs: PyTorchObs, actions: th.Tensor) -> tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """

        # Preprocess the observation if needed
        # features = self.extract_features(obs)
        obs = obs.view(obs.shape[0], self.seq_len, -1)

        if self.share_features_extractor:
            # latent_pi, latent_vf = self.mlp_extractor(obs)
            latent_pi = self.mlp_extractor.value_net(obs[:, :-1])[1][0].squeeze(0)
            latent_vf = self.mlp_extractor.policy_net(obs[:, :-1])[1][0].squeeze(0)
        else:
            pi_features, vf_features = obs
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy