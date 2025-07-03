# Copyright (c) 2022-2025, The Isaac Lab Project Developers. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration for a Humanoid Running task for Isaac Lab v2.2.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

# --- Version 2.2 Specific Imports ---
from isaaclab.managers import CommandManagerCfg, CurriculumManagerCfg, RewardManagerCfg, TerminationManagerCfg
from isaaclab.scene import SceneCfg
from isaaclab.assets.utils import ISAAC_NUCLEUS_DIR
# ---

from . import mdp

##
# Scene definition
##

@configclass
class HumanoidSceneCfg(SceneCfg):
    """Configuration for the humanoid running scene."""
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Isaac/Robots/Unitree/h1.usd"),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.05)),
        actuators={"body": sim_utils.IdealPDActuatorCfg(joint_names_expr=[".*"], stiffness=400.0, damping=40.0)},
    )
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(prim_path="/World/DomeLight", spawn=sim_utils.DomeLightCfg(intensity=500.0))

##
# MDP settings
##

@configclass
class RewardsCfg(RewardManagerCfg):
    """Reward terms for the MDP."""
    # The dictionary keys are the names of the reward terms
    tracking_lin_vel_xy_exp = {"func": mdp.track_lin_vel_xy_exp, "weight": 1.5, "params": {"std": 0.5}}
    tracking_ang_vel_z_exp = {"func": mdp.track_ang_vel_z_exp, "weight": 0.75, "params": {"std": 0.5}}
    penalty_joint_acc = {"func": mdp.joint_acc_l2, "weight": -5.0e-7}
    penalty_action_rate = {"func": mdp.action_rate_l2, "weight": -5.0e-5}
    penalty_applied_torque = {"func": mdp.applied_torque_l2, "weight": -1.0e-5}
    reward_upright = {"func": mdp.upright, "weight": 1.0}

@configclass
class TerminationsCfg(TerminationManagerCfg):
    """Termination terms for the MDP."""
    time_out = {"func": mdp.time_out, "time_out": True}
    base_contact = {"func": mdp.base_contact, "params": {"threshold": 1}}

@configclass
class CurriculumCfg(CurriculumManagerCfg):
    """Curriculum terms for the MDP."""
    ranges = { "command.base_velocity.ranges.lin_vel_x": (0.5, 3.0) }

##
# Environment configuration
##

@configclass
class HumanoidRunEnvCfg(ManagerBasedRLEnvCfg):
    # This is the correct structure for Isaac Lab v2.2
    scene: HumanoidSceneCfg = HumanoidSceneCfg(num_envs=4096, env_spacing=3.0)
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post-initialization."""
        super().__post_init__()
        # general settings
        self.decimation = 2
        self.episode_length_s = 20.0
        # command manager
        self.commands = CommandManagerCfg(
            command_cfg=sim_utils.UniformVelocityCommandCfg(
                asset_name="robot",
                resampling_time_range=(5.0, 5.0),
                rel_standing_freq=0.1,
                ranges=sim_utils.UniformVelocityCommandCfg.Ranges(
                    lin_vel_x=(0.5, 1.0),
                    lin_vel_y=(0.0, 0.0),
                    ang_vel_z=(-0.25, 0.25),
                ),
            ),
            command_name="base_velocity"
        )
        # viewer
        self.viewer.eye = (8.0, 8.0, 4.0)
