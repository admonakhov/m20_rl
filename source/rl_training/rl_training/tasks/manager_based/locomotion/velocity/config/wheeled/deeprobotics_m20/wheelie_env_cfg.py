# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp

from .flat_env_cfg import DeeproboticsM20FlatEnvCfg


@configclass
class DeeproboticsM20WheelieEnvCfg(DeeproboticsM20FlatEnvCfg):
    """M20 wheelie task: drive only on rear wheels with the front half kept out of contact."""

    joint_sdk_names = [
        "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint", "fl_wheel_joint",
        "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint", "fr_wheel_joint",
        "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint", "hl_wheel_joint",
        "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint", "hr_wheel_joint",
    ]

    all_hipy_joint_target = [-math.pi / 2.0, -math.pi / 2.0, -math.pi / 2.0, -math.pi / 2.0]

    front_leg_joint_names = [
        "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint",
        "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint",
    ]
    rear_leg_joint_names = [
        "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint",
        "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint",
    ]
    front_wheel_joint_names = ["fl_wheel_joint", "fr_wheel_joint"]
    rear_wheel_joint_names = ["hl_wheel_joint", "hr_wheel_joint"]
    leg_joint_names = front_leg_joint_names + rear_leg_joint_names
    wheel_joint_names = front_wheel_joint_names + rear_wheel_joint_names
    joint_names = leg_joint_names + wheel_joint_names
    active_joint_names = rear_leg_joint_names + rear_wheel_joint_names
    inactive_joint_names = front_leg_joint_names + front_wheel_joint_names

    base_link_name = "base_link"
    front_wheel_body_regex = "f[l,r]_wheel"
    rear_wheel_body_regex = "h[l,r]_wheel"
    wheel_body_regex = ".*_wheel"
    non_wheel_body_regex = "^(?!.*_wheel).*$"

    def __post_init__(self):
        super().__post_init__()
        self.events.randomize_rigid_body_material.params["static_friction_range"] = [0.15, 1.5]
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = [0.10, 1.2]
        # ------------------------------Robot pose------------------------------
        self.leg_joint_names = self.front_leg_joint_names + self.rear_leg_joint_names
        self.wheel_joint_names = self.front_wheel_joint_names + self.rear_wheel_joint_names
        self.joint_names = self.leg_joint_names + self.wheel_joint_names
        self.hipx_joint_names = ["fl_hipx_joint", "fr_hipx_joint", "hl_hipx_joint", "hr_hipx_joint"]
        self.hipy_joint_names = ["fl_hipy_joint", "fr_hipy_joint", "hl_hipy_joint", "hr_hipy_joint"]
        self.knee_joint_names = ["fl_knee_joint", "fr_knee_joint", "hl_knee_joint", "hr_knee_joint"]
        self.foot_link_name = self.rear_wheel_body_regex
        self.scene.robot.joint_sdk_names = self.joint_sdk_names

        self.scene.robot.init_state.pos = (0.0, 0.0, 0.52)
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.scene.robot.init_state.joint_pos = {
            "fl_hipx_joint": 0.0,
            "fl_hipy_joint": -0.6,
            "fl_knee_joint": 1.0,
            "fr_hipx_joint": 0.0,
            "fr_hipy_joint": -0.6,
            "fr_knee_joint": 1.0,
            "hl_hipx_joint": 0.0,
            "hl_hipy_joint": 0.6,
            "hl_knee_joint": -1.0,
            "hr_hipx_joint": 0.0,
            "hr_hipy_joint": 0.6,
            "hr_knee_joint": -1.0,
            "fl_wheel_joint": 0.0,
            "fr_wheel_joint": 0.0,
            "hl_wheel_joint": 0.0,
            "hr_wheel_joint": 0.0,
        }

        # ------------------------------Commands------------------------------
        self.commands.base_velocity = mdp.UniformThresholdVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(8.0, 8.0),
            rel_standing_envs=0.10,
            rel_heading_envs=1.0,
            heading_command=False,
            heading_control_stiffness=0.75,
            debug_vis=True,
            ranges=mdp.VerticalBodyVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(-0.0, 0.0),
                ang_vel_z=(-1, 1),
                # heading=(-math.pi, math.pi),
            ),
        )
        self.deploy_base_velocity_ranges = {
            "lin_vel_x": (-0.8, 0.8),
            "lin_vel_y": (-0.15, 0.15),
            "ang_vel_z": (-0.6, 0.6),
            "heading": (-math.pi, math.pi),
        }

        # ------------------------------Observations------------------------------
        self.observations.policy.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.policy.joint_pos.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=self.joint_names, preserve_order=True),
            "wheel_asset_cfg": SceneEntityCfg("robot", joint_names=self.wheel_joint_names),
        }
        self.observations.critic.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.critic.joint_pos.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=self.joint_names, preserve_order=True),
            "wheel_asset_cfg": SceneEntityCfg("robot", joint_names=self.wheel_joint_names),
        }
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].preserve_order = True
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].preserve_order = True

        # ------------------------------Actions------------------------------
        self.actions.joint_pos.scale = {".*_hipx_joint": 0.125, "^(?!.*_hipx_joint).*": 0.25}
        self.actions.joint_vel.scale = 6.0
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_vel.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.leg_joint_names
        self.actions.joint_vel.joint_names = self.wheel_joint_names

        # ------------------------------Events------------------------------
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot = None
        self.events.randomize_actuator_gains.params["asset_cfg"].joint_names = self.joint_names
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.01, 0.02),
                "roll": (-0.03, 0.03),
                "pitch": (-0.06, 0.06),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.05, 0.05),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.08, 0.08),
            },
        }

        # ------------------------------Rewards------------------------------
        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.ang_vel_xy_l2.weight = -0.10
        self.rewards.flat_orientation_l2.weight = 0.0

        self.rewards.base_height_l2.weight = -4.0
        self.rewards.base_height_l2.params["target_height"] = 0.82
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]

        self.rewards.body_lin_acc_l2.weight = -0.05
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        self.rewards.joint_torques_l2.weight = -5.0e-5
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_torques_wheel_l2.weight = -1.0e-5
        self.rewards.joint_torques_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names

        self.rewards.joint_vel_l2.weight = 0.0
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = self.leg_joint_names

        self.rewards.joint_vel_wheel_l2.weight = -5.0e-4
        self.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names

        self.rewards.joint_acc_l2.weight = -2.0e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.leg_joint_names

        self.rewards.joint_acc_wheel_l2.weight = -1.0e-7
        self.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names

        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.leg_joint_names

        self.rewards.joint_vel_limits.weight = -0.1
        self.rewards.joint_vel_limits.params["asset_cfg"].joint_names = self.wheel_joint_names

        self.rewards.joint_power.weight = -2.0e-5
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.leg_joint_names

        self.rewards.stand_still.weight = -0.5
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.rear_leg_joint_names

        self.rewards.action_rate_l2.weight = -0.005

        self.rewards.track_lin_vel_xy_exp.weight = 3.5
        self.rewards.track_lin_vel_xy_exp.params["std"] = math.sqrt(0.25)

        self.rewards.track_ang_vel_z_exp.func = mdp.track_ang_vel_z_world_exp
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.params["std"] = math.sqrt(0.20)

        self.rewards.wheel_vel_penalty.weight = -0.05
        self.rewards.wheel_vel_penalty.params["sensor_cfg"].body_names = [self.rear_wheel_body_regex]
        self.rewards.wheel_vel_penalty.params["asset_cfg"].joint_names = self.rear_wheel_joint_names

        self.rewards.undesired_contacts.weight = -2.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [self.non_wheel_body_regex]
        self.rewards.undesired_contacts.params["threshold"] = 5.0

        self.rewards.contact_forces.weight = -2.0e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.rear_wheel_body_regex]

        self.rewards.feet_contact.weight = -0.5
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.rear_wheel_body_regex]
        self.rewards.feet_contact.params["expect_contact_num"] = 2
        self.rewards.feet_contact_without_cmd.weight = 0.25
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.rear_wheel_body_regex]
        
        self.rewards.feet_stumble.weight = 0.0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.rear_wheel_body_regex]
        
        self.rewards.feet_slide.weight = -0.05
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.rear_wheel_body_regex]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.rear_wheel_body_regex]

        self.rewards.upward.weight = 0.0

        self.rewards.wheelie_body_pitch = RewTerm(
            func=mdp.body_forward_vertical_l2,
            weight=-3.0,
            params={"target": 0.90, "asset_cfg": SceneEntityCfg("robot")},
        )
        # self.rewards.front_leg_hold = RewTerm(
        #     func=mdp.joint_target_deviation_l2,
        #     weight=-2.0,
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot", joint_names=self.front_leg_joint_names),
        #         "target": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #     },
        # )
        self.rewards.front_wheel_vel_l2 = RewTerm(
            func=mdp.joint_vel_l2,
            weight=-0.2,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=self.front_wheel_joint_names)},
        )
        self.rewards.front_wheel_contacts = RewTerm(
            func=mdp.undesired_contacts,
            weight=-2.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.front_wheel_body_regex]), "threshold": 1.0},
        )
        self.rewards.front_leg_vel_l2 = RewTerm(
            func=mdp.joint_vel_l2,
            weight=-0.05,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=self.front_leg_joint_names)},
        )
        self.rewards.joint_mirror.weight = -0.03
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["fl_(hipx|hipy|knee).*", "fr_(hipx|hipy|knee).*"],
            ["hl_(hipx|hipy|knee).*", "hr_(hipx|hipy|knee).*"],
        ]

        # Disable locomotion terms that are tied to four supporting wheels.
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.feet_air_time_lin_xy.weight = 0.0
        self.rewards.feet_air_time_x_neg.weight = 0.0
        self.rewards.feet_air_time_ang_z.weight = 0.0
        self.rewards.feet_air_time_variance.weight = 0.0
        self.rewards.feet_gait.weight = 0.0
        self.rewards.phase_foot_trajectory_exp.weight = 0.0
        self.rewards.feet_height.weight = 0.0
        self.rewards.feet_height_body.weight = 0.0
        self.rewards.action_mirror.weight = 0.0
        self.rewards.action_sync.weight = 0.0

        self.rewards.hipx_joint_pos_penalty.weight = -0.2
        self.rewards.hipx_joint_pos_penalty.params["asset_cfg"].joint_names = self.hipx_joint_names
        
        self.rewards.hipy_joint_pos_penalty = RewTerm(
            func=mdp.joint_target_deviation_l2,
            weight=-0.2,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.hipy_joint_names),
                "target": self.all_hipy_joint_target,
            },
        )
        self.rewards.knee_joint_pos_penalty.weight = -0.2
        self.rewards.knee_joint_pos_penalty.params["asset_cfg"].joint_names = self.knee_joint_names

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.non_wheel_body_regex]), "threshold": 1.0},
        )
        self.terminations.bad_orientation_2 = None

        # ------------------------------Curriculums------------------------------
        # self.curriculum.command_levels = CurrTerm(
        #     func=mdp.command_levels_vel,
        #     params={
        #         "reward_term_name": "track_lin_vel_xy_exp",
        #         "range_multiplier": (0.35, 1.0),
        #         "success_fraction": 0.6,
        #     },
        # )

        if self.__class__.__name__ == "DeeproboticsM20WheelieEnvCfg":
            self.disable_zero_weight_rewards()
