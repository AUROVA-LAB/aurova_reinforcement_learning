# import gymnasium as gym

# # from .bimanual_direct_env_cfg import BimanualDirectCfg
# from . import agents


# gym.register(
#     id="Isaac-Bimanual-Direct-reach-v0",
#     entry_point=f"{__name__}.bimanual_direct_env:BimanualDirect",
#     # entry_point="isaaclab_tasks.aurova_reinforcement_learning.bimanual_handover.bimanual_direct_env:BimanualDirect",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.bimanual_direct_env:BimanualDirectCfg",
#         # "env_cfg_entry_point": BimanualDirectCfg,
#         "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
#         "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
#         "skrl_ddpg_cfg_entry_point": f"{agents.__name__}:skrl_ddpg_cfg.yaml"
#     }
# )

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Cartpole balancing environment.
"""

import gymnasium as gym


from .bimanual_direct_env_cfg import BimanualDirectCfg
from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Bimanual-Direct-reach-v0",
    entry_point=f"{__name__}.bimanual_direct_env:BimanualDirect",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bimanual_direct_env_cfg:BimanualDirectCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",

        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
        "skrl_ddpg_cfg_entry_point": f"{agents.__name__}:skrl_ddpg_cfg.yaml"
    }
)
