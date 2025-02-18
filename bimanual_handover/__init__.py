import gymnasium as gym

from .bimanual_direct_env_cfg import BimanualDirectCfg
from .bimanual_direct_env_joint_cfg import BimanualDirectJointCfg
from . import agents


gym.register(
    id="Isaac-Bimanual-Direct-reach-v0",
    entry_point="omni.isaac.lab_tasks.manager_based.classic.aurova_reinforcement_learning.bimanual_handover.bimanual_direct_env:BimanualDirect",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BimanualDirectCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
        "skrl_ddpg_cfg_entry_point": f"{agents.__name__}:skrl_ddpg_cfg.yaml"
    }
)

gym.register(
    id="Isaac-Bimanual-Direct-reach-joint-v0",
    entry_point="omni.isaac.lab_tasks.manager_based.classic.aurova_reinforcement_learning.bimanual_handover.bimanual_direct_env_joint_space:BimanualDirect",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BimanualDirectJointCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    }
)
