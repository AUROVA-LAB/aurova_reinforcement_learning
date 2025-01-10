import torch as th
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import MlpExtractor
import os
import gymnasium as gym
import copy

# Custom MlpFeatureExtractor
class CustomMlpExtractor(MlpExtractor):
    def __init__(self, feature_dim, net_arch, activation_fn, device, phase, option, prev_feature_extractor = nn.Sequential()):
        super().__init__(feature_dim, net_arch, activation_fn, device)

        # Filter by phases and options
        if phase == 1:
            
            # Uses Option 1 arquitecture
            if option == 1:

                # Feature extractor from Approach phase
                self.prev_feature_extractor = prev_feature_extractor.to(device)
                
                # Builds combined feature extractor
                last_layer = prev_feature_extractor[-2].out_features + 16
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
        ee_obj = self.prev_feature_extractor(features[:, :14])

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
        ee_obj = self.prev_feature_extractor(features[:, :14])

        # Return latent activations
        return self.value_net(th.cat((ee_obj, features[:, 14:]), dim = -1))



# Define a custom policy
class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, **kwargs):
        
        # Extract the arguments for the phase filtering
        # my_kwargs = kwargs.pop("my_kwargs")

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            **kwargs,
        )

        # # If the phase is MANIPULATION
        # if my_kwargs["phase"] == my_kwargs["MANIPULATION"]:
        #     if my_kwargs["option"] == 1:
        #         # Change the settings to create a model for approaching
        #         my_kwargs["cfg"]["policy_kwargs"]["my_kwargs"]["phase"] = my_kwargs["APPROACH"]
        #         my_kwargs["env"].action_space = gym.spaces.Box(low = -1000, high = 1000, shape = (6,))
        #         my_kwargs["env"].observation_space = gym.spaces.Box(low = -1000, high = 1000, shape = (7+7,))

        #         # Creates model for approaching
        #         model = PPO(
        #             policy=CustomActorCriticPolicy,
        #             env = my_kwargs["env"],  # Replace with your environment
        #             verbose=1,
        #             **my_kwargs["cfg"]
        #         )

        #         # Reset the phase
        #         my_kwargs["cfg"]["policy_kwargs"]["my_kwargs"]["phase"] = my_kwargs["MANIPULATION"]

        #         # Load pre-trained weights
        #         pretrained_path = os.path.join("/workspace/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/classic/aurova_reinforcement_learning/bimanual_handover/train/logs/", my_kwargs["path"], "policy.pth")
        #         weights = th.load(pretrained_path, map_location="cpu", weights_only = "True")

        #         # Load weights into the model
        #         model.policy.load_state_dict(weights)

        #         # Obtain the pre-trained feature extractor
        #         prev_feature_extractor = model.policy.mlp_extractor.policy_net

        #         # Freeze the model
        #         for name, p in prev_feature_extractor.named_parameters():
        #             p.requires_grad = False
                
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