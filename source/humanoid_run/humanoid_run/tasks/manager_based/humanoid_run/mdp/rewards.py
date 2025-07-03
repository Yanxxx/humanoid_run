# mdp/rewards.py

import torch
from isaaclab.envs import ManagerBasedRLEnv

##
# Task-specific rewards
##

def track_lin_vel_xy_exp(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """Reward for tracking linear velocity commands in the xy plane."""
    lin_vel_error = torch.sum(torch.square(env.commands[:, 0:2] - env.scene.robot.data.root_lin_vel_b[:, 0:2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)

def track_ang_vel_z_exp(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """Reward for tracking angular velocity commands in the z direction."""
    ang_vel_error = torch.square(env.commands[:, 2] - env.scene.robot.data.root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / std**2)

##
# Stability and style rewards
##

def upright(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Encourage the robot to stay upright."""
    # 1.0 when upright, 0.0 when perpendicular to z-axis
    return env.scene.robot.data.projected_gravity_b[:, 2]

##
# Effort and energy penalties
##

def joint_acc_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large joint accelerations."""
    return torch.sum(torch.square(env.scene.robot.data.joint_acc), dim=1)

def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large changes in action from one step to the next."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)

def applied_torque_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the amount of torque applied to the joints."""
    return torch.sum(torch.square(env.scene.robot.data.applied_torque), dim=1)
