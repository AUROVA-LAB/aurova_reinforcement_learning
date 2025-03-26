"""Configuration for the UR5e robot"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from math import pi
## 
# Configuration
##

GEN3_4f_CFG = ArticulationCfg(
    spawn = sim_utils.UsdFileCfg(
        usd_path="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/aurova_reinforcement_learning/rl_manipulation/config/usd/gen3_4f.usd",
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
            velocity_limit_sim = 100.0,
            effort_limit_sim = 87.0,
            stiffness = 800.0,
            damping = 40.0,
        ),
        "hand": ImplicitActuatorCfg(
            joint_names_expr=[".*_0"],
            velocity_limit_sim = 100.0,
            effort_limit_sim = 0.5,
            stiffness = 3.0,
            damping = 0.1,
            friction = 0.01,
        ), 
    },

)

UR5e_4f_CFG = ArticulationCfg(
    spawn = sim_utils.UsdFileCfg(
        usd_path="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/aurova_reinforcement_learning/rl_manipulation/config/usd/ur5e_4f_ros2.usd",
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
            velocity_limit_sim = 100.0,
            effort_limit_sim = 87.0,
            stiffness = 800.0,
            damping = 40.0,
        ),
        "hand": ImplicitActuatorCfg(
            joint_names_expr=[".*_0"],
            velocity_limit_sim = 100.0,
            effort_limit_sim = 0.5,
            stiffness = 3.0,
            damping = 0.1,
            friction = 0.01,
        ), 
    },

)

UR5e_3f_CFG = ArticulationCfg(
    spawn = sim_utils.UsdFileCfg(
        usd_path="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/aurova_reinforcement_learning/rl_manipulation/config/usd/ur5e_3f_ros2.usd",
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
            velocity_limit_sim = 100.0,
            effort_limit_sim = 87.0,
            stiffness = 800.0,
            damping = 40.0,
        ),
        "hand": ImplicitActuatorCfg(
            joint_names_expr=["robotiq.*"],
            velocity_limit_sim = 100.0,
            effort_limit_sim = 0.5,
            stiffness = 3.0,
            damping = 0.1,
            friction = 0.01,
        ), 
    },

)


UR5e_NOGRIP_CFG = ArticulationCfg(
    spawn = sim_utils.UsdFileCfg(
        usd_path="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/aurova_reinforcement_learning/rl_manipulation/config/usd/ur5e_ros2_grippless.usd",
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
            velocity_limit_sim = 100.0,
            effort_limit_sim = 87.0,
            stiffness = 800.0,
            damping = 40.0,
        ),
    },

)
