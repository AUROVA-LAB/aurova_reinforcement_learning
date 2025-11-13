import torch as th
import torch.nn as nn
import warnings
from typing import Optional, Union
import copy

from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    MlpExtractor
)

from stable_baselines3.common.utils import (
    get_device,
)




class customMlpExtractor(MlpExtractor):
    """
    Constructs an MLP that receives the output from a previous features extractor (i.e. a CNN) or directly
    the observations (if no features extractor is applied) as an input and outputs a latent representation
    for the policy and a value network.

    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers.
    It can be in either of the following forms:
    1. ``dict(vf=[<list of layer sizes>], pi=[<list of layer sizes>])``: to specify the amount and size of the layers in the
        policy and value nets individually. If it is missing any of the keys (pi or vf),
        zero layers will be considered for that key.
    2. ``[<list of layer sizes>]``: "shortcut" in case the amount and size of the layers
        in the policy and value nets are the same. Same as ``dict(vf=int_list, pi=int_list)``
        where int_list is the same for the actor and critic.

    .. note::
        If a key is not specified or an empty list is passed ``[]``, a linear network will be used.

    :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
    :param net_arch: The specification of the policy and value networks.
        See above for details on its formatting.
    :param activation_fn: The activation function to use for the networks.
    :param device: PyTorch device.
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: Union[list[int], dict[str, list[int]]],
        activation_fn: type[nn.Module],
        device: Union[th.device, str] = "auto",
    ) -> None:
        super().__init__(feature_dim, net_arch, activation_fn)
        device = get_device(device)

        # # save dimensions of layers in policy and value nets
        # if isinstance(net_arch, dict):
        #     # Note: if key is not specified, assume linear network
        #     pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
        #     vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
        # else:
        #     pi_layers_dims = vf_layers_dims = net_arch
        # # Iterate through the policy layers and build the policy net
        # for curr_layer_dim in pi_layers_dims:
        #     policy_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
        #     policy_net.append(activation_fn())
        #     last_layer_dim_pi = curr_layer_dim
        # # Iterate through the value layers and build the value net
        # for curr_layer_dim in vf_layers_dims:
        #     value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
        #     value_net.append(activation_fn())
        #     last_layer_dim_vf = curr_layer_dim

        self.policy_net_img = th.nn.Sequential(
            th.nn.Conv2d(1, 16, kernel_size = 3),
            th.nn.BatchNorm2d(16),
            th.nn.Tanh(),

            th.nn.Conv2d(16, 32, kernel_size = 3),
            th.nn.BatchNorm2d(32),
            th.nn.Tanh(),

            th.nn.Conv2d(32, 64, kernel_size = 3),
            th.nn.BatchNorm2d(64),
            th.nn.Tanh(),

            # th.nn.Conv2d(64, 128, kernel_size = 3),
            # th.nn.BatchNorm2d(128),
            # th.nn.Tanh(),

            th.nn.Flatten(),

            th.nn.Linear(350464, 256),
            th.nn.LayerNorm(256),
            th.nn.Tanh(),
        )

        self.value_net_img = copy.deepcopy(self.policy_net_img)

        self.policy_net = th.nn.Sequential(
            th.nn.Linear(7, 128),
            th.nn.LayerNorm(128),
            th.nn.Tanh(),

            th.nn.Linear(128, 128),
            th.nn.LayerNorm(128),
            th.nn.Tanh(),
        )
        self.value_net = copy.deepcopy(self.policy_net)

        # Save dim, used to create the distributions
        self.latent_dim_pi = feature_dim
        self.latent_dim_vf = feature_dim

        # # Create networks
        # # If the list of layers is empty, the network will just act as an Identity module
        # self.policy_net = nn.Sequential(*policy_net).to(device)
        # self.value_net = nn.Sequential(*value_net).to(device)

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """

        return self.forward_actor(features), self.forward_critic(features)
    

    def extract_features(self, features:th.Tensor) -> th.Tensor:
        geom_obs = features[:, :7]
        img_obs = features[:, 7:].reshape(-1, 1, 80, 80)

        return geom_obs, img_obs

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        geom_obs, img_obs = self.extract_features(features)

        latent_pi_geom = self.policy_net(geom_obs)
        latent_pi_img = self.policy_net_img(img_obs)


        return th.cat((latent_pi_geom, latent_pi_img), dim = 1)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        
        geom_obs, img_obs = self.extract_features(features)

        latent_vf_geom = self.value_net(geom_obs)
        latent_vf_img = self.value_net_img(img_obs)

        return th.cat((latent_vf_geom, latent_vf_img), dim = 1)






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

        self.mlp_extractor = customMlpExtractor(feature_dim = 256 + 128,
                                                net_arch=net_arch,
                                                activation_fn=nn.Tanh,
                                                device = self.device)


        # Determine the number of input features
        features = 256+128

        # Replace the actor with your custom network
        self.action_net = th.nn.Sequential(
            th.nn.Linear(features, 7),
            th.nn.Tanh())
        
        # Replace the actor with your custom network
        self.value_net = th.nn.Sequential(
            th.nn.Linear(features, 1),
            th.nn.Tanh())

        # Reinitialize parameters (important)
        self.action_net.apply(self.init_weights)
        self.value_net.apply(self.init_weights)

    
    
    def forward(self, obs: th.Tensor, deterministic: bool = False) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        
        latent_pi, latent_vf = self.mlp_extractor(obs)


        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob


    def extract_features(  # type: ignore[override]
        self, obs: PyTorchObs, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> Union[th.Tensor, tuple[th.Tensor, th.Tensor]]:
        """
        Preprocess the observation if needed and extract features.

        :param obs: Observation
        :param features_extractor: The features extractor to use. If None, then ``self.features_extractor`` is used.
        :return: The extracted features. If features extractor is not shared, returns a tuple with the
            features for the actor and the features for the critic.
        """
        if self.share_features_extractor:
            return super().extract_features(obs, self.features_extractor if features_extractor is None else features_extractor)
        else:
            if features_extractor is not None:
                warnings.warn(
                    "Provided features_extractor will be ignored because the features extractor is not shared.",
                    UserWarning,
                )

            pi_features = super().extract_features(obs, self.pi_features_extractor)
            vf_features = super().extract_features(obs, self.vf_features_extractor)
            return pi_features, vf_features

