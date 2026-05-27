/**
 * @file rl_control_state.hpp
 * @brief rl policy runnning state for quadruped-wheel robot
 * @author DeepRobotics
 * @version 1.0
 * @date 2025-11-07
 * 
 * @copyright Copyright (c) 2025  DeepRobotics
 * 
 */
#pragma once
#include "state_base.h"
#include "policy_runner_base.hpp"
#include "m20_policy_runner.hpp"
#include "robot_interface.h"
#include "user_command_interface.h"
#include "json.hpp"
#include "basic_function.hpp"
#include "yaml-cpp/yaml.h"

#include <cstdlib>
#include <filesystem>

namespace qw {
    class RLControlState : public StateBase {
    private:
        struct BackFlipFsmConfig {
            bool enabled = false;
            StateName target_state = StateName::kLieDown;
            float projected_gravity_z_threshold = 0.2f;
            float recovery_projected_gravity_z_threshold = -0.1f;
            float min_duration = 0.15f;
        };

        RobotBasicState rbs_[2];
        std::atomic<int> rbs_write_index_{0};
        int getrbsReadIndex() const { return 1 - rbs_write_index_.load(std::memory_order_acquire); }

        int state_run_cnt_;

        std::shared_ptr<PolicyRunnerBase> policy_ptr_;
        std::shared_ptr<M20PolicyRunner> m20_policy_;

        std::thread run_policy_thread_;
        bool start_flag_ = true;

        float policy_cost_time_ = 1;
        std::filesystem::path model_path_;
        BackFlipFsmConfig back_flip_fsm_cfg_;
        double back_flip_detected_since_ = -1.0;
        bool back_flip_switch_requested_ = false;

        Eigen::MatrixXf acc_rot = Eigen::MatrixXf::Zero(20, 3);
        int acc_rot_count = 0;

        void UpdateRobotObservation() {
            int write_idx = rbs_write_index_.load(std::memory_order_relaxed);
            RobotBasicState& buffer = rbs_[write_idx];

            buffer.base_rpy = ri_ptr_->GetImuRpy();
            buffer.base_rot_mat = RpyToRm(buffer.base_rpy);
            buffer.base_omega = ri_ptr_->GetImuOmega();
            buffer.base_acc = ri_ptr_->GetImuAcc();
            buffer.joint_pos = ri_ptr_->GetJointPosition();
            buffer.joint_vel = ri_ptr_->GetJointVelocity();
            buffer.joint_tau = ri_ptr_->GetJointTorque();

            // 储存
            buffer.flt_base_acc_mat.row(acc_rot_count) = buffer.base_acc.transpose();
            acc_rot_count += 1;
            acc_rot_count = acc_rot_count % 20;

            rbs_write_index_.store(1 - write_idx,  std::memory_order_release);
        }

        std::filesystem::path ResolveDeployConfigPath() const {
            if (const char* env_path = std::getenv("M20_DEPLOY_CFG")) {
                const std::filesystem::path candidate(env_path);
                if (std::filesystem::exists(candidate)) {
                    return candidate;
                }
            }

            const std::vector<std::filesystem::path> candidates = {
                model_path_.parent_path() / "deploy.yaml",
                model_path_.parent_path() / "params" / "deploy.yaml",
                model_path_.parent_path().parent_path() / "params" / "deploy.yaml",
                model_path_.parent_path() / "_deploy.yaml",
            };

            for (const auto& candidate : candidates) {
                if (std::filesystem::exists(candidate)) {
                    return candidate;
                }
            }
            return {};
        }

        static StateName ParseStateName(const std::string& state_name) {
            if (state_name == "liedown" || state_name == "lie_down" || state_name == "kLieDown") {
                return StateName::kLieDown;
            }
            if (state_name == "joint_damping" || state_name == "kJointDamping") {
                return StateName::kJointDamping;
            }
            if (state_name == "standup" || state_name == "stand_up" || state_name == "kStandUp") {
                return StateName::kStandUp;
            }
            if (state_name == "rl_control" || state_name == "kRLControl") {
                return StateName::kRLControl;
            }
            throw std::runtime_error("Unknown FSM target_state in deploy.yaml: " + state_name);
        }

        void LoadBackFlipFsmConfig() {
            const auto deploy_path = ResolveDeployConfigPath();
            if (deploy_path.empty()) {
                return;
            }

            const YAML::Node root = YAML::LoadFile(deploy_path.string());
            const YAML::Node cfg = root["fsm"]["auto_switch_on_back"];
            if (!cfg) {
                return;
            }

            back_flip_fsm_cfg_.enabled = cfg["enabled"] ? cfg["enabled"].as<bool>() : back_flip_fsm_cfg_.enabled;
            if (cfg["target_state"]) {
                back_flip_fsm_cfg_.target_state = ParseStateName(cfg["target_state"].as<std::string>());
            }
            if (cfg["projected_gravity_z_threshold"]) {
                back_flip_fsm_cfg_.projected_gravity_z_threshold = cfg["projected_gravity_z_threshold"].as<float>();
            }
            if (cfg["recovery_projected_gravity_z_threshold"]) {
                back_flip_fsm_cfg_.recovery_projected_gravity_z_threshold =
                    cfg["recovery_projected_gravity_z_threshold"].as<float>();
            }
            if (cfg["min_duration"]) {
                back_flip_fsm_cfg_.min_duration = cfg["min_duration"].as<float>();
            }
        }

        void UpdateBackFlipFsmSwitch() {
            if (!back_flip_fsm_cfg_.enabled) {
                return;
            }

            const RobotBasicState& robot_obs = rbs_[getrbsReadIndex()];
            const float projected_gravity_z = (robot_obs.base_rot_mat.transpose() * Vec3f(0.0f, 0.0f, -1.0f))(2);
            const double now = ri_ptr_->GetInterfaceTimeStamp();

            if (projected_gravity_z >= back_flip_fsm_cfg_.projected_gravity_z_threshold) {
                if (back_flip_detected_since_ < 0.0) {
                    back_flip_detected_since_ = now;
                }
                if (now - back_flip_detected_since_ >= back_flip_fsm_cfg_.min_duration) {
                    back_flip_switch_requested_ = true;
                }
                return;
            }

            if (projected_gravity_z <= back_flip_fsm_cfg_.recovery_projected_gravity_z_threshold) {
                back_flip_detected_since_ = -1.0;
                back_flip_switch_requested_ = false;
            }
        }

        void PolicyRunner() {
            int run_cnt_record = -1;
            while (start_flag_) {
                if (state_run_cnt_ % policy_ptr_->decimation_ == 0 && state_run_cnt_ != run_cnt_record) {
                    timespec start_timestamp, end_timestamp;
                    clock_gettime(CLOCK_MONOTONIC, &start_timestamp);
                    auto ra = policy_ptr_->getRobotAction(rbs_[getrbsReadIndex()], *(uc_ptr_->GetUserCommand()));
                    
                    MatXf res = ra.ConvertToMat();

                    ri_ptr_->SetJointCommand(res);
                    run_cnt_record = state_run_cnt_;
                    clock_gettime(CLOCK_MONOTONIC, &end_timestamp);
                    policy_cost_time_ = (end_timestamp.tv_sec - start_timestamp.tv_sec) * 1e3
                                        + (end_timestamp.tv_nsec - start_timestamp.tv_nsec) / 1e6;

                }
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }

    public:
        RLControlState(const RobotName &robot_name, const std::string &state_name,
                       std::shared_ptr<ControllerData> data_ptr) : StateBase(robot_name, state_name, data_ptr) {
            if (robot_name_ == RobotName::M20) {
                namespace fs = std::filesystem;
                fs::path base = fs::path(__FILE__).parent_path();
                model_path_ = fs::canonical(base / ".." / ".." / "policy" / "policy.onnx");
                m20_policy_ = std::make_shared<M20PolicyRunner>("m20_policy", model_path_.string());
                LoadBackFlipFsmConfig();
            }

            policy_ptr_ = m20_policy_;
            if (!policy_ptr_) {
                std::cerr << "error policy" << std::endl;
                exit(0);
            }
            policy_ptr_->DisplayPolicyInfo();
        }

        ~RLControlState() {}

        virtual void OnEnter() {
            state_run_cnt_ = -1;
            start_flag_ = true;
            back_flip_detected_since_ = -1.0;
            back_flip_switch_requested_ = false;
            run_policy_thread_ = std::thread(std::bind(&RLControlState::PolicyRunner, this));
            policy_ptr_->OnEnter();
            StateBase::msfb_.UpdateCurrentState(RobotMotionState::RLControlMode);
        };

        virtual void OnExit() {
            start_flag_ = false;
            run_policy_thread_.join();
            state_run_cnt_ = -1;
        }

        virtual void Run() {
            UpdateRobotObservation();
            UpdateBackFlipFsmSwitch();
            state_run_cnt_++;
        }

        virtual bool LoseControlJudge() {
            if (uc_ptr_->GetUserCommand()->target_mode == uint8_t(RobotMotionState::JointDamping)) return true;
            return PostureUnsafeCheck();
        }

        bool PostureUnsafeCheck() {
            // Vec3f rpy = ri_ptr_->GetImuRpy();
            // if(rpy(0) > 30./180*M_PI || rpy(1) > 45./180*M_PI){
            //     std::cout << "posture value: " << 180./M_PI*rpy.transpose() << std::endl;
            //     return true;
            // }
            return false;
        }

        virtual StateName GetNextStateName() {
            if (uc_ptr_->GetUserCommand()->safe_control_mode != 0) 
                return StateName::kJointDamping;
            if (back_flip_switch_requested_)
                return back_flip_fsm_cfg_.target_state;
            if (uc_ptr_->GetUserCommand()->target_mode == uint8_t(RobotMotionState::LieDown))
                return StateName::kLieDown;
            
            return StateName::kRLControl;
        }
    };
};
