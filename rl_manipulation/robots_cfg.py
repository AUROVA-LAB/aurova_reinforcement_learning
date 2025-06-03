"""Configuration for the UR5e robot"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg
from math import pi
## 
# Configuration
##

GEN3_4f_CFG = ArticulationCfg(
    spawn = sim_utils.UsdFileCfg(
        usd_path="/workspace/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/classic/aurova_reinforcement_learning/rl_manipulation/config/usd/gen3_4f.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "arm_joint_1": 0.0,
            "arm_joint_2": -pi/8,
            "arm_joint_3": 0.0,
            "arm_joint_4": 3*pi/4,
            "arm_joint_5": 0.0,
            "arm_joint_6": -pi/6,
            "arm_joint_7": 5*pi/4 - pi,
            "joint_12_0": 0.263,
        },
        pos = (-1.25, 0.0, 0.0)
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr = ["arm_.*"],
            velocity_limit = 100.0,
            effort_limit = 87.0,
            stiffness = 800.0,
            damping = 40.0,
        ),
        "hand": ImplicitActuatorCfg(
            joint_names_expr=[".*_0"],
            velocity_limit = 100.0,
            effort_limit = 0.5,
            stiffness = 3.0,
            damping = 0.1,
            friction = 0.01,
        ), 
    },

)

UR5e_4f_CFG = ArticulationCfg(
    spawn = sim_utils.UsdFileCfg(
        usd_path="/workspace/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/classic/aurova_reinforcement_learning/rl_manipulation/config/usd/ur5e_4f_ros2.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "arm_shoulder_pan_joint": 0.0,
            "arm_shoulder_lift_joint": -pi/2,
            "arm_elbow_joint": -pi/2,
            "arm_wrist_1_joint": 0.0,
            "arm_wrist_2_joint": pi/2,
            "arm_wrist_3_joint": pi/4,
            "joint_0_0": 0.0,
            "joint_1_0": 0.263*6,
            "joint_2_0": 0.263*5,
            "joint_3_0": 0.263*2.3,
            "joint_4_0": 0.0,
            "joint_5_0": 0.263*6,
            "joint_6_0": 0.263*5,
            "joint_7_0": 0.263*2.3,
            "joint_8_0": 0.0,
            "joint_9_0": 0.263*6,
            "joint_10_0": 0.263*5,
            "joint_11_0": 0.263*2.3,
            "joint_12_0": 0.263, # zero position is 0.263
            "joint_13_0": 0.0,
            "joint_14_0": 0.263*5,
            "joint_15_0": 0.263*5,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr = ["arm_.*"],
            velocity_limit = 100.0,
            effort_limit = 87.0,
            stiffness = 800.0,
            damping = 40.0,
        ),
        "hand": ImplicitActuatorCfg(
            joint_names_expr=[".*_0"],
            velocity_limit = 100.0,
            effort_limit = 0.5,
            stiffness = 3.0,
            damping = 0.1,
            friction = 0.01,
        ), 
    },

)

UR5e_3f_CFG = ArticulationCfg(
    spawn = sim_utils.UsdFileCfg(
        usd_path="/workspace/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/classic/aurova_reinforcement_learning/rl_manipulation/config/usd/ur5e_position_ros2_3f.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -pi/2,
            "elbow_joint": -pi/2,
            "wrist_1_joint": -pi/2,
            "wrist_2_joint": pi/2,
            "wrist_3_joint": -pi/4,
            "robotiq_finger_middle_joint_1": 0.049,
            "robotiq_finger_1_joint_1": 0.049,
            "robotiq_finger_2_joint_1": 0.049,
            "robotiq_finger_middle_joint_3": -0.052,
            "robotiq_finger_1_joint_3": -0.052,
            "robotiq_finger_2_joint_3": -0.052,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                                "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
            velocity_limit = 100.0,
            effort_limit = 87.0,
            stiffness = 800.0,
            damping = 40.0,
        ),
        "hand": ImplicitActuatorCfg(
            joint_names_expr=["robotiq.*"],
            velocity_limit = 100.0,
            effort_limit = 0.5,
            stiffness = 3.0,
            damping = 0.1,
            friction = 0.01,
        ), 
    },

)


UR5e_NOGRIP_CFG = ArticulationCfg(
    spawn = sim_utils.UsdFileCfg(
        usd_path="/workspace/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/classic/aurova_reinforcement_learning/rl_manipulation/config/usd/ur5e_ros2_grippless.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -pi/2,
            "elbow_joint": -pi/2,
            "wrist_1_joint": -pi/2,
            "wrist_2_joint": pi/2,
            "wrist_3_joint": -pi/4,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                                "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
            velocity_limit = 100.0,
            effort_limit = 87.0,
            stiffness = 800.0,
            damping = 40.0,
        ),
    },

    # Best params:  {'distance': tensor(34.8215, device='cuda:0'), 
    #           'damping': tensor([49.0432, 36.6490, 60.8989,  1.2644,  7.0457,  6.1012], device='cuda:0'), 
    #           'stiffness': tensor([2255.1541,  597.0469,  656.7890,    6.1423,  444.1543,  161.5248], device='cuda:0')}
    
    # actuators={
        
    #     "shoulder_pan": ImplicitActuatorCfg(
    #         joint_names_expr = ["shoulder_pan_joint"],
    #         velocity_limit = 100.0,
    #         effort_limit = 87.0,
    #         stiffness = 2255.1541,# 1375.3032, # 800.0,
    #         damping = 49.0432# 39.7259 #40.0,
    #     ),
    #     "shoiulder_lift": ImplicitActuatorCfg(
    #         joint_names_expr = ["shoulder_lift_joint"],
    #         velocity_limit = 100.0,
    #         effort_limit = 87.0,
    #         stiffness = 597.0469, # 1114.5306, 
    #         damping = 36.6490# 29.7188,
    #     ),
    #     "elbow": ImplicitActuatorCfg(
    #         joint_names_expr = ["elbow_joint"],
    #         velocity_limit = 100.0,
    #         effort_limit = 87.0,
    #         stiffness = 656.7890, #565.5358, 
    #         damping = 60.8989# 105.8098,
    #     ),
    #     "wrist_1": ImplicitActuatorCfg(
    #         joint_names_expr = ["wrist_1_joint"],
    #         velocity_limit = 100.0,
    #         effort_limit = 87.0,
    #         stiffness = 6.1423, #9.4776,
    #         damping = 1.2644# 1.0331
    #     ),
    #     "wrist_2": ImplicitActuatorCfg(
    #         joint_names_expr = ["wrist_2_joint"],
    #         velocity_limit = 100.0,
    #         effort_limit = 87.0,
    #         stiffness = 444.1543, #384.5534,
    #         damping = 7.0457# 8.2002
    #     ),
    #     "wrist_3": ImplicitActuatorCfg(
    #         joint_names_expr = ["wrist_3_joint"],
    #         velocity_limit = 100.0,
    #         effort_limit = 87.0,
    #         stiffness = 161.5248, #89.3253,
    #         damping = 6.1012# 7.4152,
    #     ),
        

    # },
)
