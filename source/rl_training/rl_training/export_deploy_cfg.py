import numpy as np
import os
import yaml
import re

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import class_to_dict


def format_value(x):
    if isinstance(x, float):
        return float(f"{x:.3g}")
    elif isinstance(x, list):
        return [format_value(i) for i in x]
    elif isinstance(x, dict):
        return {k: format_value(v) for k, v in x.items()}
    else:
        return x


def export_deploy_cfg(env: ManagerBasedRLEnv, log_dir):
    asset: Articulation = env.scene["robot"]
    available_joint_names = list(asset.data.joint_names)
    joint_sdk_names = getattr(env.cfg.scene.robot, "joint_sdk_names", None)
    if joint_sdk_names is None:
        joint_sdk_names = getattr(env.cfg, "joint_names", None)
    if joint_sdk_names is None:
        joint_sdk_names = available_joint_names

    resolved_joint_names = []
    resolved_joint_asset_ids = []
    for joint_name in joint_sdk_names:
        if joint_name in available_joint_names:
            resolved_joint_names.append(joint_name)
            resolved_joint_asset_ids.append(available_joint_names.index(joint_name))
            continue

        matches = [name for name in available_joint_names if re.fullmatch(joint_name, name)]
        if len(matches) == 1:
            resolved_joint_names.append(matches[0])
            resolved_joint_asset_ids.append(available_joint_names.index(matches[0]))

    if not resolved_joint_names:
        resolved_joint_names = available_joint_names
        resolved_joint_asset_ids = list(range(len(available_joint_names)))

    cfg = {}  # noqa: SIM904
    cfg["joint_names"] = resolved_joint_names
    cfg["step_dt"] = env.cfg.sim.dt * env.cfg.decimation
    stiffness = asset.data.default_joint_stiffness[0].detach().cpu().numpy()[resolved_joint_asset_ids]
    cfg["stiffness"] = stiffness.tolist()
    damping = asset.data.default_joint_damping[0].detach().cpu().numpy()[resolved_joint_asset_ids]
    cfg["damping"] = damping.tolist()
    default_joint_pos = asset.data.default_joint_pos[0].detach().cpu().numpy()[resolved_joint_asset_ids]
    cfg["default_joint_pos"] = default_joint_pos.tolist()

    # --- commands ---
    cfg["commands"] = {}
    if hasattr(env.cfg.commands, "base_velocity"):  # some environments do not have base_velocity command
        cfg["commands"]["base_velocity"] = {}
        deploy_ranges = getattr(env.cfg, "deploy_base_velocity_ranges", None)
        if deploy_ranges is not None:
            ranges = {key: list(value) for key, value in deploy_ranges.items()}
        elif hasattr(env.cfg.commands.base_velocity, "limit_ranges"):
            ranges = env.cfg.commands.base_velocity.limit_ranges.to_dict()
        else:
            ranges = env.cfg.commands.base_velocity.ranges.to_dict()
        for item_name in ["lin_vel_x", "lin_vel_y", "ang_vel_z"]:
            if item_name in ranges:
                ranges[item_name] = list(ranges[item_name])
        cfg["commands"]["base_velocity"]["ranges"] = ranges

    # --- actions ---
    action_names = env.action_manager.active_terms
    action_terms = zip(action_names, env.action_manager._terms.values())
    cfg["actions"] = {}
    for action_name, action_term in action_terms:
        term_cfg = action_term.cfg.copy()
        if isinstance(term_cfg.scale, float):
            term_cfg.scale = [term_cfg.scale for _ in range(action_term.action_dim)]
        else:  # dict
            term_cfg.scale = action_term._scale[0].detach().cpu().numpy().tolist()

        if term_cfg.clip is not None:
            term_cfg.clip = action_term._clip[0].detach().cpu().numpy().tolist()

        if action_name in ["JointPositionAction", "JointVelocityAction"]:
            if term_cfg.use_default_offset:
                term_cfg.offset = action_term._offset[0].detach().cpu().numpy().tolist()
            else:
                term_cfg.offset = [0.0 for _ in range(action_term.action_dim)]

        # clean cfg
        term_cfg = term_cfg.to_dict()

        for _ in ["class_type", "asset_name", "debug_vis", "preserve_order", "use_default_offset"]:
            del term_cfg[_]
        cfg["actions"][action_name] = term_cfg

        if action_term._joint_ids == slice(None):
            cfg["actions"][action_name]["joint_ids"] = None
        else:
            cfg["actions"][action_name]["joint_ids"] = action_term._joint_ids

    policy_order = []
    if "joint_pos" in cfg["actions"] and "joint_vel" in cfg["actions"]:
        policy_order.extend(cfg["actions"]["joint_pos"].get("joint_names", []))
        policy_order.extend(cfg["actions"]["joint_vel"].get("joint_names", []))
    cfg["joint_ids_map"] = [resolved_joint_names.index(joint_name) for joint_name in policy_order]

    # --- observations ---
    obs_names = env.observation_manager.active_terms["policy"]
    obs_cfgs = env.observation_manager._group_obs_term_cfgs["policy"]
    obs_terms = zip(obs_names, obs_cfgs)
    cfg["observations"] = {}
    for obs_name, obs_cfg in obs_terms:
        obs_dims = tuple(obs_cfg.func(env, **obs_cfg.params).shape)
        term_cfg = obs_cfg.copy()
        if term_cfg.scale is not None:
            scale = term_cfg.scale.detach().cpu().numpy().tolist()
            if isinstance(scale, float):
                term_cfg.scale = [scale for _ in range(obs_dims[1])]
            else:
                term_cfg.scale = scale
        else:
            term_cfg.scale = [1.0 for _ in range(obs_dims[1])]
        if term_cfg.clip is not None:
            term_cfg.clip = list(term_cfg.clip)
        if term_cfg.history_length == 0:
            term_cfg.history_length = 1

        # clean cfg
        term_cfg = term_cfg.to_dict()
        for _ in ["func", "modifiers", "noise", "flatten_history_dim"]:
            del term_cfg[_]
        cfg["observations"][obs_name] = term_cfg

    # --- deploy-time FSM helpers ---
    cfg["fsm"] = {
        "auto_switch_on_back": {
            "enabled": True,
            "target_state": "liedown",
            "projected_gravity_z_threshold": 0.2,
            "recovery_projected_gravity_z_threshold": -0.1,
            "min_duration": 0.15,
        }
    }

    # --- save config file ---
    filename = os.path.join(log_dir, "params", "deploy.yaml")
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not isinstance(cfg, dict):
        cfg = class_to_dict(cfg)
    cfg = format_value(cfg)
    with open(filename, "w") as f:
        yaml.dump(cfg, f, default_flow_style=None, sort_keys=False)
