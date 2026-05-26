#pragma once

#include "policy_runner_base.hpp"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "yaml-cpp/yaml.h"

class M20PolicyRunner : public PolicyRunnerBase {
private:
    static constexpr int kRobotMotorNum = 16;
    static constexpr float kControlLoopDt = 0.005f;

    struct ObservationTermConfig {
        std::string name;
        std::vector<float> scale;
        std::vector<int> joint_ids;
        std::vector<int> zero_joint_ids;
        int history_length = 1;
    };

    struct DeployConfig {
        std::filesystem::path path;
        std::vector<std::string> policy_order;
        std::vector<float> action_scale;
        std::vector<float> default_joint_pos;
        std::vector<float> stiffness;
        std::vector<float> damping;
        std::vector<ObservationTermConfig> observation_terms;
        Vec3f max_cmd_vel = Vec3f::Zero();
        float step_dt = 0.02f;
        int action_pos_count = 12;
        int action_vel_count = 4;

        DeployConfig() {
            max_cmd_vel << 2.0f, 1.0f, 1.0f;
        }
    };

    inline static const std::array<std::string, kRobotMotorNum> kRobotJointNames = {
        "fl_hipx_joint",  "fl_hipy_joint",  "fl_knee_joint",  "fl_wheel_joint",
        "fr_hipx_joint",  "fr_hipy_joint",  "fr_knee_joint",  "fr_wheel_joint",
        "hl_hipx_joint",  "hl_hipy_joint",  "hl_knee_joint",  "hl_wheel_joint",
        "hr_hipx_joint",  "hr_hipy_joint",  "hr_knee_joint",  "hr_wheel_joint",
    };

    Ort::Env ort_env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;

    std::string model_path_;
    std::string input_name_ = "obs";
    std::string output_name_ = "actions";
    std::string deploy_cfg_path_;

    std::array<int64_t, 2> input_observation_shape_{1, 57};
    std::vector<int> policy_to_robot_joint_ids_;
    std::vector<ObservationTermConfig> observation_terms_;

    VecXf action_scale_policy_;
    VecXf default_joint_pos_policy_;
    VecXf kp_policy_;
    VecXf kd_policy_;

    VecXf default_joint_pos_robot_;
    VecXf kp_robot_;
    VecXf kd_robot_;
    VecXf last_action_eigen_;

    Vec3f max_cmd_vel_;
    float agent_timestep_ = 0.02f;
    int observation_dim_ = 57;
    int action_dim_ = 16;
    int action_pos_count_ = 12;
    int action_vel_count_ = 4;

public:
    M20PolicyRunner(const std::string& policy_name, const std::string& model_path)
        : PolicyRunnerBase(policy_name),
          ort_env_(ORT_LOGGING_LEVEL_WARNING, policy_name.c_str()),
          memory_info_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)),
          model_path_(model_path) {
        ConfigureLegacyFallback();
        TryLoadDeployConfig();
        ConfigureSession();
    }

    void DisplayPolicyInfo() override {
        std::cout << "[Policy] " << policy_name_ << "\n";
        std::cout << "  model: " << model_path_ << "\n";
        std::cout << "  deploy: " << (deploy_cfg_path_.empty() ? "<legacy defaults>" : deploy_cfg_path_) << "\n";
        std::cout << "  obs_dim: " << observation_dim_ << ", action_dim: " << action_dim_ << "\n";
        std::cout << "  policy_dt: " << agent_timestep_ << "s, decimation: " << decimation_ << std::endl;
    }

    void OnEnter() override {
        run_cnt_ = 0;
        cmd_vel_input_.setZero();
        last_action_eigen_ = VecXf::Zero(action_dim_);
    }

    RobotAction getRobotAction(const RobotBasicState& robot_obs, const UserCommand& user_cmd) override {
        VecXf obs = BuildObservation(robot_obs, user_cmd);
        VecXf action = RunPolicy(obs);
        last_action_eigen_ = action;
        ++run_cnt_;
        return BuildRobotAction(action);
    }

private:
    static std::unordered_map<std::string, int> MakeJointNameToRobotIdMap() {
        std::unordered_map<std::string, int> index_by_name;
        for (int i = 0; i < kRobotMotorNum; ++i) {
            index_by_name.emplace(kRobotJointNames.at(i), i);
        }
        return index_by_name;
    }

    static VecXf MakeLegacyDefaultJointPos() {
        VecXf values(kRobotMotorNum);
        values << 0.0f, -0.6f, 1.0f, 0.0f,
                  0.0f, -0.6f, 1.0f, 0.0f,
                  0.0f,  0.6f, -1.0f, 0.0f,
                  0.0f,  0.6f, -1.0f, 0.0f;
        return values;
    }

    static VecXf MakeLegacyStiffness() {
        VecXf values(kRobotMotorNum);
        values << 80.0f, 80.0f, 80.0f, 0.0f,
                  80.0f, 80.0f, 80.0f, 0.0f,
                  80.0f, 80.0f, 80.0f, 0.0f,
                  80.0f, 80.0f, 80.0f, 0.0f;
        return values;
    }

    static VecXf MakeLegacyDamping() {
        VecXf values(kRobotMotorNum);
        values << 2.0f, 2.0f, 2.0f, 0.6f,
                  2.0f, 2.0f, 2.0f, 0.6f,
                  2.0f, 2.0f, 2.0f, 0.6f,
                  2.0f, 2.0f, 2.0f, 0.6f;
        return values;
    }

    static std::vector<int> MakeLegacyPolicyToRobotJointIds() {
        return {0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 3, 7, 11, 15};
    }

    void ConfigureLegacyFallback() {
        default_joint_pos_robot_ = MakeLegacyDefaultJointPos();
        kp_robot_ = MakeLegacyStiffness();
        kd_robot_ = MakeLegacyDamping();

        max_cmd_vel_ << 2.0f, 1.0f, 1.0f;
        agent_timestep_ = 0.02f;
        observation_dim_ = 57;
        action_dim_ = 16;
        action_pos_count_ = 12;
        action_vel_count_ = 4;
        decimation_ = 4;
        run_cnt_ = 0;
        deploy_cfg_path_.clear();

        policy_to_robot_joint_ids_ = MakeLegacyPolicyToRobotJointIds();

        action_scale_policy_ = VecXf(kRobotMotorNum);
        action_scale_policy_ << 0.125f, 0.25f, 0.25f,
                                0.125f, 0.25f, 0.25f,
                                0.125f, 0.25f, 0.25f,
                                0.125f, 0.25f, 0.25f,
                                5.0f, 5.0f, 5.0f, 5.0f;

        default_joint_pos_policy_ = VecXf(kRobotMotorNum);
        kp_policy_ = VecXf(kRobotMotorNum);
        kd_policy_ = VecXf(kRobotMotorNum);
        for (int i = 0; i < kRobotMotorNum; ++i) {
            const int robot_joint_id = policy_to_robot_joint_ids_.at(i);
            default_joint_pos_policy_(i) = default_joint_pos_robot_(robot_joint_id);
            kp_policy_(i) = kp_robot_(robot_joint_id);
            kd_policy_(i) = kd_robot_(robot_joint_id);
        }
        last_action_eigen_ = VecXf::Zero(action_dim_);

        observation_terms_.clear();
        observation_terms_.push_back({"base_ang_vel", {0.25f, 0.25f, 0.25f}, {}, {}, 1});
        observation_terms_.push_back({"projected_gravity", {1.0f, 1.0f, 1.0f}, {}, {}, 1});
        observation_terms_.push_back({"velocity_commands", {1.0f, 1.0f, 1.0f}, {}, {}, 1});

        ObservationTermConfig joint_pos_term;
        joint_pos_term.name = "joint_pos";
        joint_pos_term.scale = std::vector<float>(kRobotMotorNum, 1.0f);
        joint_pos_term.joint_ids = policy_to_robot_joint_ids_;
        joint_pos_term.zero_joint_ids = {3, 7, 11, 15};
        observation_terms_.push_back(joint_pos_term);

        ObservationTermConfig joint_vel_term;
        joint_vel_term.name = "joint_vel";
        joint_vel_term.scale = std::vector<float>(kRobotMotorNum, 0.05f);
        joint_vel_term.joint_ids = policy_to_robot_joint_ids_;
        observation_terms_.push_back(joint_vel_term);

        observation_terms_.push_back({"actions", std::vector<float>(kRobotMotorNum, 1.0f), {}, {}, 1});
        input_observation_shape_ = {1, observation_dim_};
    }

    void ConfigureSession() {
        session_options_.SetIntraOpNumThreads(1);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        session_ = std::make_unique<Ort::Session>(ort_env_, model_path_.c_str(), session_options_);

        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name_holder = session_->GetInputNameAllocated(0, allocator);
        auto output_name_holder = session_->GetOutputNameAllocated(0, allocator);
        input_name_ = input_name_holder.get();
        output_name_ = output_name_holder.get();

        const auto input_shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (input_shape.size() != 2) {
            throw std::runtime_error("Expected a 2D policy input tensor.");
        }
        if (input_shape[1] > 0 && input_shape[1] != observation_dim_) {
            throw std::runtime_error(
                "deploy.yaml observation size (" + std::to_string(observation_dim_) +
                ") does not match ONNX input size (" + std::to_string(input_shape[1]) + ").");
        }

        const auto output_shape = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (output_shape.size() != 2) {
            throw std::runtime_error("Expected a 2D policy output tensor.");
        }
        if (output_shape[1] > 0 && output_shape[1] != action_dim_) {
            throw std::runtime_error(
                "deploy.yaml action size (" + std::to_string(action_dim_) +
                ") does not match ONNX output size (" + std::to_string(output_shape[1]) + ").");
        }
    }

    void TryLoadDeployConfig() {
        try {
            const std::filesystem::path deploy_path = ResolveDeployConfigPath();
            if (deploy_path.empty()) {
                std::cout << "[Policy] deploy.yaml not found, using legacy defaults.\n";
                return;
            }
            ApplyDeployConfig(ParseDeployConfig(deploy_path));
        } catch (const std::exception& e) {
            std::cerr << "[Policy] Failed to load deploy config: " << e.what()
                      << "\n[Policy] Falling back to legacy defaults." << std::endl;
            ConfigureLegacyFallback();
        }
    }

    std::filesystem::path ResolveDeployConfigPath() const {
        if (const char* env_path = std::getenv("M20_DEPLOY_CFG")) {
            const std::filesystem::path candidate(env_path);
            if (std::filesystem::exists(candidate)) {
                return candidate;
            }
        }

        const std::filesystem::path model_path(model_path_);
        const std::vector<std::filesystem::path> candidates = {
            model_path.parent_path() / "deploy.yaml",
            model_path.parent_path() / "params" / "deploy.yaml",
            model_path.parent_path().parent_path() / "params" / "deploy.yaml",
            model_path.parent_path() / "_deploy.yaml",
        };

        for (const auto& candidate : candidates) {
            if (std::filesystem::exists(candidate)) {
                return candidate;
            }
        }
        return {};
    }

    static std::vector<float> LoadFloatSequence(const YAML::Node& node) {
        std::vector<float> values;
        if (!node) {
            return values;
        }
        if (node.IsSequence()) {
            values.reserve(node.size());
            for (const auto& value : node) {
                values.push_back(value.as<float>());
            }
        } else if (node.IsScalar()) {
            values.push_back(node.as<float>());
        } else {
            throw std::runtime_error("Expected a float scalar or sequence in deploy.yaml.");
        }
        return values;
    }

    static std::vector<std::string> LoadStringSequence(const YAML::Node& node) {
        std::vector<std::string> values;
        if (!node) {
            return values;
        }
        if (!node.IsSequence()) {
            throw std::runtime_error("Expected a string sequence in deploy.yaml.");
        }
        values.reserve(node.size());
        for (const auto& value : node) {
            values.push_back(value.as<std::string>());
        }
        return values;
    }

    static std::vector<int> JointNamesToRobotIds(
        const std::vector<std::string>& joint_names,
        const std::unordered_map<std::string, int>& robot_joint_map) {
        std::vector<int> joint_ids;
        joint_ids.reserve(joint_names.size());
        for (const auto& joint_name : joint_names) {
            const auto it = robot_joint_map.find(joint_name);
            if (it == robot_joint_map.end()) {
                throw std::runtime_error("Unknown robot joint in deploy.yaml: " + joint_name);
            }
            joint_ids.push_back(it->second);
        }
        return joint_ids;
    }

    ObservationTermConfig ParseObservationTerm(
        const std::string& term_name,
        const YAML::Node& term_node,
        const std::unordered_map<std::string, int>& robot_joint_map) const {
        ObservationTermConfig term;
        term.name = term_name;
        term.history_length = term_node["history_length"] ? term_node["history_length"].as<int>() : 1;
        if (term.history_length != 1) {
            throw std::runtime_error("Only history_length=1 is supported in sdk_deploy.");
        }

        if (term_name == "joint_pos" || term_name == "joint_vel") {
            const YAML::Node asset_cfg = term_node["params"]["asset_cfg"];
            const auto joint_names = LoadStringSequence(asset_cfg["joint_names"]);
            term.joint_ids = JointNamesToRobotIds(joint_names, robot_joint_map);

            if (term_name == "joint_pos") {
                const YAML::Node wheel_cfg = term_node["params"]["wheel_asset_cfg"];
                if (wheel_cfg && wheel_cfg["joint_names"]) {
                    term.zero_joint_ids =
                        JointNamesToRobotIds(LoadStringSequence(wheel_cfg["joint_names"]), robot_joint_map);
                }
            }
        }

        term.scale = LoadFloatSequence(term_node["scale"]);
        if ((term_name == "joint_pos" || term_name == "joint_vel") && term.scale.size() != term.joint_ids.size()) {
            throw std::runtime_error("Observation scale size does not match joint list for term: " + term_name);
        }
        return term;
    }

    DeployConfig ParseDeployConfig(const std::filesystem::path& deploy_path) const {
        const YAML::Node root = YAML::LoadFile(deploy_path.string());
        const auto robot_joint_map = MakeJointNameToRobotIdMap();

        if (!root["actions"] || !root["actions"]["joint_pos"] || !root["actions"]["joint_vel"]) {
            throw std::runtime_error("deploy.yaml is missing required action terms.");
        }

        DeployConfig cfg;
        cfg.path = deploy_path;
        cfg.step_dt = root["step_dt"] ? root["step_dt"].as<float>() : 0.02f;

        if (root["commands"] && root["commands"]["base_velocity"] && root["commands"]["base_velocity"]["ranges"]) {
            const YAML::Node ranges = root["commands"]["base_velocity"]["ranges"];
            auto max_abs = [](const YAML::Node& range_node, float fallback) {
                if (!range_node || !range_node.IsSequence() || range_node.size() != 2) {
                    return fallback;
                }
                return std::max(std::abs(range_node[0].as<float>()), std::abs(range_node[1].as<float>()));
            };
            cfg.max_cmd_vel << max_abs(ranges["lin_vel_x"], 2.0f),
                               max_abs(ranges["lin_vel_y"], 1.0f),
                               max_abs(ranges["ang_vel_z"], 1.0f);
        }

        const YAML::Node pos_action = root["actions"]["joint_pos"];
        const YAML::Node vel_action = root["actions"]["joint_vel"];

        const auto pos_joint_names = LoadStringSequence(pos_action["joint_names"]);
        const auto vel_joint_names = LoadStringSequence(vel_action["joint_names"]);
        const auto pos_scale = LoadFloatSequence(pos_action["scale"]);
        const auto vel_scale = LoadFloatSequence(vel_action["scale"]);

        cfg.action_pos_count = static_cast<int>(pos_joint_names.size());
        cfg.action_vel_count = static_cast<int>(vel_joint_names.size());
        cfg.policy_order = pos_joint_names;
        cfg.policy_order.insert(cfg.policy_order.end(), vel_joint_names.begin(), vel_joint_names.end());
        cfg.action_scale = pos_scale;
        cfg.action_scale.insert(cfg.action_scale.end(), vel_scale.begin(), vel_scale.end());

        cfg.default_joint_pos = LoadFloatSequence(root["default_joint_pos"]);
        cfg.stiffness = LoadFloatSequence(root["stiffness"]);
        cfg.damping = LoadFloatSequence(root["damping"]);

        const std::size_t action_dim = cfg.policy_order.size();
        if (cfg.action_scale.size() != action_dim ||
            cfg.default_joint_pos.size() != action_dim ||
            cfg.stiffness.size() != action_dim ||
            cfg.damping.size() != action_dim) {
            throw std::runtime_error("Action-related vectors in deploy.yaml must match policy action order.");
        }

        if (!root["observations"]) {
            throw std::runtime_error("deploy.yaml is missing observations.");
        }
        for (const auto& item : root["observations"]) {
            cfg.observation_terms.push_back(ParseObservationTerm(item.first.as<std::string>(), item.second, robot_joint_map));
        }
        return cfg;
    }

    void ApplyDeployConfig(const DeployConfig& cfg) {
        const auto robot_joint_map = MakeJointNameToRobotIdMap();
        const std::vector<int> policy_robot_ids = JointNamesToRobotIds(cfg.policy_order, robot_joint_map);

        action_dim_ = static_cast<int>(cfg.policy_order.size());
        action_pos_count_ = cfg.action_pos_count;
        action_vel_count_ = cfg.action_vel_count;
        agent_timestep_ = cfg.step_dt;
        max_cmd_vel_ = cfg.max_cmd_vel;
        decimation_ = std::max(1, static_cast<int>(std::lround(agent_timestep_ / kControlLoopDt)));

        policy_to_robot_joint_ids_ = policy_robot_ids;
        observation_terms_ = cfg.observation_terms;

        default_joint_pos_robot_ = MakeLegacyDefaultJointPos();
        kp_robot_ = MakeLegacyStiffness();
        kd_robot_ = MakeLegacyDamping();

        action_scale_policy_ = VecXf::Zero(action_dim_);
        default_joint_pos_policy_ = VecXf::Zero(action_dim_);
        kp_policy_ = VecXf::Zero(action_dim_);
        kd_policy_ = VecXf::Zero(action_dim_);

        for (int i = 0; i < action_dim_; ++i) {
            const int robot_joint_id = policy_to_robot_joint_ids_.at(i);
            action_scale_policy_(i) = cfg.action_scale.at(i);
            default_joint_pos_policy_(i) = cfg.default_joint_pos.at(i);
            kp_policy_(i) = cfg.stiffness.at(i);
            kd_policy_(i) = cfg.damping.at(i);

            default_joint_pos_robot_(robot_joint_id) = cfg.default_joint_pos.at(i);
            kp_robot_(robot_joint_id) = cfg.stiffness.at(i);
            kd_robot_(robot_joint_id) = cfg.damping.at(i);
        }

        observation_dim_ = 0;
        for (const auto& term : observation_terms_) {
            if ((term.name == "base_ang_vel" || term.name == "projected_gravity" || term.name == "velocity_commands") &&
                term.scale.size() != 3) {
                throw std::runtime_error("Observation term '" + term.name + "' must have exactly 3 values.");
            }
            if (term.name == "actions" && static_cast<int>(term.scale.size()) != action_dim_) {
                throw std::runtime_error("Observation term 'actions' must match policy action dimension.");
            }
            observation_dim_ += static_cast<int>(term.scale.size());
        }
        input_observation_shape_ = {1, observation_dim_};
        last_action_eigen_ = VecXf::Zero(action_dim_);
        deploy_cfg_path_ = cfg.path.string();
    }

    void UpdateCommandInput(const UserCommand& user_cmd) {
        Vec3f desired_cmd;
        desired_cmd << user_cmd.forward_vel_scale * max_cmd_vel_(0),
                       user_cmd.side_vel_scale * max_cmd_vel_(1),
                       user_cmd.turnning_vel_scale * max_cmd_vel_(2);

        for (int i = 0; i < 3; ++i) {
            const float delta = desired_cmd(i) - cmd_vel_input_(i);
            cmd_vel_input_(i) += LimitNumber(delta, vel_delta_const_(i));
            cmd_vel_input_(i) = LimitNumber(cmd_vel_input_(i), -max_cmd_vel_(i), max_cmd_vel_(i));
        }
    }

    Vec3f ComputeProjectedGravity(const RobotBasicState& robot_obs) const {
        return robot_obs.base_rot_mat.transpose() * Vec3f(0.0f, 0.0f, -1.0f);
    }

    VecXf BuildObservation(const RobotBasicState& robot_obs, const UserCommand& user_cmd) {
        UpdateCommandInput(user_cmd);

        VecXf obs(observation_dim_);
        int cursor = 0;
        for (const auto& term : observation_terms_) {
            if (term.name == "base_ang_vel") {
                for (std::size_t i = 0; i < term.scale.size(); ++i) {
                    obs(cursor++) = robot_obs.base_omega(static_cast<int>(i)) * term.scale.at(i);
                }
                continue;
            }

            if (term.name == "projected_gravity") {
                const Vec3f projected_gravity = ComputeProjectedGravity(robot_obs);
                for (std::size_t i = 0; i < term.scale.size(); ++i) {
                    obs(cursor++) = projected_gravity(static_cast<int>(i)) * term.scale.at(i);
                }
                continue;
            }

            if (term.name == "velocity_commands") {
                for (std::size_t i = 0; i < term.scale.size(); ++i) {
                    obs(cursor++) = cmd_vel_input_(static_cast<int>(i)) * term.scale.at(i);
                }
                continue;
            }

            if (term.name == "joint_pos") {
                const std::unordered_set<int> zero_joint_ids(term.zero_joint_ids.begin(), term.zero_joint_ids.end());
                for (std::size_t i = 0; i < term.joint_ids.size(); ++i) {
                    const int joint_id = term.joint_ids.at(i);
                    float value = robot_obs.joint_pos(joint_id) - default_joint_pos_robot_(joint_id);
                    if (zero_joint_ids.count(joint_id) > 0) {
                        value = 0.0f;
                    }
                    obs(cursor++) = value * term.scale.at(i);
                }
                continue;
            }

            if (term.name == "joint_vel") {
                for (std::size_t i = 0; i < term.joint_ids.size(); ++i) {
                    obs(cursor++) = robot_obs.joint_vel(term.joint_ids.at(i)) * term.scale.at(i);
                }
                continue;
            }

            if (term.name == "actions") {
                for (std::size_t i = 0; i < term.scale.size(); ++i) {
                    obs(cursor++) = last_action_eigen_(static_cast<int>(i)) * term.scale.at(i);
                }
                continue;
            }

            throw std::runtime_error("Unsupported observation term in deploy.yaml: " + term.name);
        }
        if (cursor != observation_dim_) {
            throw std::runtime_error("Built observation size does not match configured observation dimension.");
        }
        return obs;
    }

    VecXf RunPolicy(const VecXf& obs) {
        std::vector<float> obs_buffer(obs.data(), obs.data() + obs.size());
        const char* input_names[] = {input_name_.c_str()};
        const char* output_names[] = {output_name_.c_str()};

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_,
            obs_buffer.data(),
            obs_buffer.size(),
            input_observation_shape_.data(),
            input_observation_shape_.size());

        auto output_tensors =
            session_->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

        if (output_tensors.empty()) {
            throw std::runtime_error("Policy inference returned no output tensors.");
        }

        auto output_info = output_tensors.front().GetTensorTypeAndShapeInfo();
        const std::size_t output_size = output_info.GetElementCount();
        if (static_cast<int>(output_size) != action_dim_) {
            throw std::runtime_error(
                "Policy output size (" + std::to_string(output_size) +
                ") does not match deploy action size (" + std::to_string(action_dim_) + ").");
        }

        const float* output_data = output_tensors.front().GetTensorData<float>();
        VecXf action(action_dim_);
        for (int i = 0; i < action_dim_; ++i) {
            action(i) = output_data[i];
        }
        return action;
    }

    RobotAction BuildRobotAction(const VecXf& policy_action) const {
        RobotAction robot_action;
        robot_action.goal_joint_pos = default_joint_pos_robot_;
        robot_action.goal_joint_vel = VecXf::Zero(kRobotMotorNum);
        robot_action.kp = kp_robot_;
        robot_action.kd = kd_robot_;
        robot_action.tau_ff = VecXf::Zero(kRobotMotorNum);

        for (int i = 0; i < action_dim_; ++i) {
            const int robot_joint_id = policy_to_robot_joint_ids_.at(i);
            if (i < action_pos_count_) {
                robot_action.goal_joint_pos(robot_joint_id) =
                    default_joint_pos_policy_(i) + policy_action(i) * action_scale_policy_(i);
            } else {
                robot_action.goal_joint_vel(robot_joint_id) = policy_action(i) * action_scale_policy_(i);
            }
        }
        return robot_action;
    }
};
