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
        usd_path="/workspace/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/classic/aurova_reinforcement_learning/bimanual_handover/config/usd/gen3_4f.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint_1": 0.0,
            "joint_2": -pi/8,
            "joint_3": 0.0,
            "joint_4": 3*pi/4,
            "joint_5": 0.0,
            "joint_6": -pi/6,
            "joint_7": 5*pi/4 - pi,
            "joint_12_0": 0.263,
        },
        pos = (-1.5, 0.0, 0.0)
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },

)

UR5e_4f_CFG = ArticulationCfg(
    spawn = sim_utils.UsdFileCfg(
        usd_path="/workspace/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/classic/aurova_reinforcement_learning/bimanual_handover/config/usd/ur5e_4f_ros2.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -pi/2,
            "elbow_joint": -pi/2,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": pi/2,
            "wrist_3_joint": pi/4,
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
            joint_names_expr=[".*"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },

)

