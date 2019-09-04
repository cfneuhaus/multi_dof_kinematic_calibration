#include "multi_dof_kinematic_calibration/Calibrator.h"

#include <Eigen/Geometry>

#include "visual_marker_mapping/CameraUtilities.h"
#include "visual_marker_mapping/DetectionIO.h"
#include "visual_marker_mapping/EigenCVConversions.h"
#include "visual_marker_mapping/PropertyTreeUtilities.h"
#include "visual_marker_mapping/ReconstructionIO.h"

#include "multi_dof_kinematic_calibration/CeresUtil.h"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>


#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/version.h>

#include "multi_dof_kinematic_calibration/DebugVis.h"
#include <fstream>
#include <iostream>

template <typename Map1, typename Map2, typename F>
void iterateMatches(const Map1& m1, const Map2& m2, F&& f)
{
    if (m1.size() < m2.size())
    {
        for (auto it1 = std::begin(m1); it1 != std::end(m1); ++it1)
        {
            const auto it2 = m2.find(it1->first);
            if (it2 == m2.end())
                continue;
            f(it1->first, it1->second, it2->second);
        }
    }
    else
    {
        for (auto it2 = std::begin(m2); it2 != std::end(m2); ++it2)
        {
            const auto it1 = m1.find(it2->first);
            if (it1 == m1.end())
                continue;
            f(it1->first, it1->second, it2->second);
        }
    }
}

namespace multi_dof_kinematic_calibration
{
struct TransformationChain
{
    enum class TransformType
    {
        POSE,
        JOINT1DOF
    };
    void addPose() { chain.push_back(TransformType::POSE); }
    void add1DOFJoint() { chain.push_back(TransformType::JOINT1DOF); }

    template <typename CostFn> void addParametersToCostFn(CostFn* cost_function) const
    {
        for (const TransformType& t : chain)
        {
            if (t == TransformType::JOINT1DOF)
            {
                cost_function->AddParameterBlock(3); // trans
                cost_function->AddParameterBlock(4); // quat
                cost_function->AddParameterBlock(1); // angle
                cost_function->AddParameterBlock(1); // joint scale
            }
            else if (t == TransformType::POSE)
            {
                cost_function->AddParameterBlock(3); // trans
                cost_function->AddParameterBlock(4); // quat
            }
        }
    }

    template <typename T> Eigen::Matrix<T, 7, 1> endEffectorPose(T const* const* parameters) const
    {
        Eigen::Matrix<T, 7, 1> world_to_end_pose;
        world_to_end_pose << T(0), T(0), T(0), T(1), T(0), T(0), T(0);
        if (chain.empty())
        {
            return world_to_end_pose;
        }
        int pindex = 0;
        for (size_t i = 0; i < chain.size(); ++i)
        {
            if (chain[i] == TransformType::JOINT1DOF)
            {
                const auto rotaxis_to_parent
                    = cmakePose<T>(eMap3(parameters[pindex + 0]), eMap4(parameters[pindex + 1]));
                const T jointScale = *parameters[pindex + 3];
                const T angle = *parameters[pindex + 2] * jointScale;
                Eigen::Matrix<T, 7, 1> invRot;
                invRot << T(0), T(0), T(0), cos(-angle / T(2)), T(0), T(0), sin(-angle / T(2));

                // world_to_end_pose = cposeAdd(world_to_cam, cposeAdd(invRot, rotaxis_to_parent));

                world_to_end_pose = cposeAdd(
                    invRot, cposeAdd(rotaxis_to_parent,
                                world_to_end_pose)); // cposeAdd(world_to_cam, cposeAdd(invRot,
                // rotaxis_to_parent));

                pindex += 4;
            }
            else if (chain[i] == TransformType::POSE)
            {
                const auto parent_to_pose
                    = cmakePose<T>(eMap3(parameters[pindex + 0]), eMap4(parameters[pindex + 1]));

                world_to_end_pose = cposeAdd(parent_to_pose, world_to_end_pose);

                pindex += 2;
            }
        }
        return world_to_end_pose;
    }

    std::vector<TransformType> chain;
};

struct KinematicChainRepError
{
    KinematicChainRepError(const Eigen::Vector2d& observation, const Eigen::Vector3d& point_3d,
        const Eigen::Matrix<double, 5, 1>& d, const Eigen::Matrix3d& K,
        const TransformationChain& chain)
        : repError(observation, d, K)
        , point_3d(point_3d)
        , chain(chain)
    {
    }

    template <typename T> bool operator()(T const* const* parameters, T* residuals) const
    {
        const auto world_to_cam = chain.endEffectorPose<T>(parameters);

        const Eigen::Matrix<T, 3, 1> point_3d_T = point_3d.cast<T>();
        return repError(&world_to_cam(0), &world_to_cam(3), &point_3d_T(0), residuals);
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const Eigen::Vector2d& observation,
        const Eigen::Vector3d& point_3d, const Eigen::Matrix<double, 5, 1>& d,
        const Eigen::Matrix3d& K, const TransformationChain& chain)
    {
        auto cost_function = new ceres::DynamicAutoDiffCostFunction<KinematicChainRepError, 4>(
            new KinematicChainRepError(observation, point_3d, d, K, chain));
        chain.addParametersToCostFn(cost_function);
        cost_function->SetNumResiduals(2);
        return cost_function;
    }

    OpenCVReprojectionError repError;
    Eigen::Vector3d point_3d;
    TransformationChain chain;
};

struct KinematicChainPoseError
{
    KinematicChainPoseError(
        const Eigen::Matrix<double, 7, 1>& expected_world_to_cam, const TransformationChain& chain)
        : expected_world_to_cam(expected_world_to_cam)
        , chain(chain)
    {
    }

    template <typename T> bool operator()(T const* const* parameters, T* residuals) const
    {
        const auto world_to_cam = chain.endEffectorPose<T>(parameters);
        eMap6(&residuals[0]) = cposeManifoldMinus<T>(world_to_cam, expected_world_to_cam.cast<T>());
        // eMap3(&residuals[3])*=T(1000);
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(
        const Eigen::Matrix<double, 7, 1>& expected_world_to_cam, const TransformationChain& chain)
    {
        auto cost_function = new ceres::DynamicAutoDiffCostFunction<KinematicChainPoseError, 4>(
            new KinematicChainPoseError(expected_world_to_cam, chain));
        chain.addParametersToCostFn(cost_function);
        cost_function->SetNumResiduals(6);
        return cost_function;
    }

    Eigen::Matrix<double, 7, 1> expected_world_to_cam;
    TransformationChain chain;
};

struct MarkerPoint2PlaneError
{
    MarkerPoint2PlaneError(const Eigen::Matrix<double, 7, 1>& marker_to_world,
        const Eigen::Vector3d& laser_point, const TransformationChain& chain)
        : marker_to_world(marker_to_world)
        , laser_point(laser_point)
        , chain(chain)
    {
        marker_plane_n = cposeTransformPoint(marker_to_world, Eigen::Vector3d(0, 0, 1))
            - marker_to_world.segment<3>(0);
        marker_plane_n /= marker_plane_n.norm();
        marker_plane_d = marker_plane_n.transpose() * marker_to_world.segment<3>(0);

        //		std::cout << "pos: " << marker_to_world.transpose() << std::endl;
        //		std::cout << "norm: " << marker_plane_n.transpose() << std::endl;
    }


    template <typename T> bool operator()(T const* const* parameters, T* residuals) const
    {
        const auto world_to_laser = chain.endEffectorPose<T>(parameters);
        auto laser_to_world = cposeInv(world_to_laser);

        Eigen::Matrix<T, 3, 1> laser_point_in_world
            = cposeTransformPoint<T>(laser_to_world, laser_point.cast<T>());

        T td = laser_point_in_world.transpose() * marker_plane_n.cast<T>();

        *residuals = (T(marker_plane_d) - td) / T(0.01);
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const Eigen::Matrix<double, 7, 1>& marker_to_world,
        const Eigen::Vector3d& laser_point, const TransformationChain& chain)
    {
        auto cost_function = new ceres::DynamicAutoDiffCostFunction<MarkerPoint2PlaneError, 4>(
            new MarkerPoint2PlaneError(marker_to_world, laser_point, chain));
        chain.addParametersToCostFn(cost_function);
        cost_function->SetNumResiduals(1);
        return cost_function;
    }

    Eigen::Matrix<double, 7, 1> marker_to_world;
    Eigen::Vector3d laser_point;

    Eigen::Vector3d marker_plane_n;
    double marker_plane_d;

    TransformationChain chain;
};


//-----------------------------------------------------------------------------
Calibrator::Calibrator(CalibrationData calib_data)
    : calib_data(std::move(calib_data))
{
}
//-----------------------------------------------------------------------------
void Calibrator::optimizeUpToJoint(const std::set<size_t>& optimization_set, OptimizationMode mode)
{
    if (mode == OptimizationMode::SIMPLE_THEN_FULL)
        std::cout << "Performing simple, then full optimization" << std::endl;
    else if (mode == OptimizationMode::ONLY_SIMPLE)
        std::cout << "Performing simple optimization only" << std::endl;
    else if (mode == OptimizationMode::ONLY_FULL)
        std::cout << "Performing full optimization only" << std::endl;

    //    auto poseInverse = [](const Eigen::Matrix<double, 7, 1>& pose)
    //    {
    //        return cposeInv<double>(pose);
    //    };
    //    auto poseAdd = [](const Eigen::Matrix<double, 7, 1>& x,
    //        const Eigen::Matrix<double, 7, 1>& d) -> Eigen::Matrix<double, 7, 1>
    //    {
    //        return cposeAdd<double>(x, d);
    //    };

    auto pathFromRootToJoint = [this](const std::string& start) -> std::vector<size_t> {
        std::string cur_joint_name = start;
        std::vector<size_t> joint_list;
        if (cur_joint_name == "base")
            return joint_list;
        size_t cur_index = calib_data.name_to_joint[cur_joint_name];
        joint_list.push_back(cur_index);
        while (1)
        {
            cur_joint_name = calib_data.joints[cur_index].parent;
            if (cur_joint_name == "base")
                break;
            cur_index = calib_data.name_to_joint[cur_joint_name];
            joint_list.push_back(cur_index);
        }
        std::reverse(joint_list.begin(), joint_list.end());
        return joint_list;
    };

    std::set<size_t> parent_joint_set
        = optimization_set; // pathFromRootToJoint(calib_data.joints[upTojointIndex].name);
    std::set<size_t> descendant_joints;

    bool descendants_contain_1DOF_joint = false;

    {
        for (const auto& sensor_id_to_parent_joint : calib_data.sensor_id_to_parent_joint)
        {
            const auto path_to_cam = pathFromRootToJoint(sensor_id_to_parent_joint.second);
            // if (std::find(path_to_cam.begin(), path_to_cam.end(), upTojointIndex)
            //    != path_to_cam.end())
            {
                for (auto it = path_to_cam.rbegin(); it != path_to_cam.rend(); ++it)
                {
                    // if (*it == upTojointIndex)
                    if (parent_joint_set.count(*it))
                        break;
                    descendant_joints.insert(*it);
                    if (calib_data.joints[*it].type == "1_dof_joint")
                        descendants_contain_1DOF_joint = true;
                }
            }
        }
    }
    if (!descendants_contain_1DOF_joint)
    {
        parent_joint_set.clear();
        for (size_t i = 0; i < calib_data.joints.size(); i++)
            parent_joint_set.insert(i);
        descendant_joints.clear();
    }

    const std::vector<size_t> parent_joints(parent_joint_set.begin(), parent_joint_set.end());

    if (!descendant_joints.empty())
    {
        std::cout << "Replacing " << descendant_joints.size() << " joints with temp poses: ";
        for (auto d : descendant_joints)
            std::cout << calib_data.joints[d].name << ", ";
        std::cout << std::endl;
    }

    ceres::Problem problem_simple;
    ceres::Problem problem_full;

    const std::vector<int> yzconstant_params = { 1, 2 };
    const auto yzconstant_parametrization = new ceres::SubsetParameterization(3, yzconstant_params);
    const auto yzconstant_parametrization2
        = new ceres::SubsetParameterization(3, yzconstant_params);

    const auto quaternion_parameterization = new ceres::QuaternionParameterization;
    const auto quaternion_parameterization2 = new ceres::QuaternionParameterization;

    for (size_t pj = 0; pj < parent_joints.size(); pj++)
    {
        const size_t j = parent_joints[pj];

        auto& parent_to_joint_pose = jointData[j].parent_to_joint_pose;

        if (calib_data.joints[j].type == "1_dof_joint")
        {
            // x always has to be positive; y,z have to be 0
            problem_simple.AddParameterBlock(
                &parent_to_joint_pose(0), 3, yzconstant_parametrization);
            problem_simple.AddParameterBlock(
                &parent_to_joint_pose(3), 4, quaternion_parameterization);

            // problem_simple.SetParameterLowerBound(&joint_to_parent_pose(0), 0, 0);

            problem_simple.AddParameterBlock(&jointData[j].ticks_to_rad, 1);
            // problem_simple.SetParameterBlockConstant(&jointData[j].ticks_to_rad);

            problem_full.AddParameterBlock(
                &parent_to_joint_pose(0), 3, yzconstant_parametrization2);
            problem_full.AddParameterBlock(
                &parent_to_joint_pose(3), 4, quaternion_parameterization2);

            problem_full.AddParameterBlock(&jointData[j].ticks_to_rad, 1);
            // problem_full.SetParameterBlockConstant(&jointData[j].ticks_to_rad);

            // problem_full.SetParameterLowerBound(&joint_to_parent_pose(0), 0, 0);

            if (calib_data.joints[j].fixed)
            {
                problem_simple.SetParameterBlockConstant(&parent_to_joint_pose(0));
                problem_simple.SetParameterBlockConstant(&parent_to_joint_pose(3));
                problem_full.SetParameterBlockConstant(&parent_to_joint_pose(0));
                problem_full.SetParameterBlockConstant(&parent_to_joint_pose(3));
            }
        }
        else if (calib_data.joints[j].type == "pose")
        {
            problem_simple.AddParameterBlock(&parent_to_joint_pose(0), 3);
            problem_simple.AddParameterBlock(
                &parent_to_joint_pose(3), 4, quaternion_parameterization);

            problem_full.AddParameterBlock(&parent_to_joint_pose(0), 3);
            problem_full.AddParameterBlock(
                &parent_to_joint_pose(3), 4, quaternion_parameterization2);


            if (calib_data.joints[j].fixed)
            {
                problem_simple.SetParameterBlockConstant(&parent_to_joint_pose(0));
                problem_simple.SetParameterBlockConstant(&parent_to_joint_pose(3));
                problem_full.SetParameterBlockConstant(&parent_to_joint_pose(0));
                problem_full.SetParameterBlockConstant(&parent_to_joint_pose(3));
            }
        }
    }


    using DistinctJointIndex = std::vector<size_t>;
    using DistinctLeverIndex = size_t;
    std::vector<DistinctJointIndex> indexToDistinctJoint(calib_data.calib_frames.size());
    std::vector<DistinctLeverIndex> indexToDistinctLever(calib_data.calib_frames.size());

    using JointState = int;
    using LeverState = std::vector<int>;
    std::vector<std::map<JointState, size_t> > distinctJointPositions(calib_data.joints.size());
    std::map<LeverState, DistinctLeverIndex> distinctLeverPositions;

    std::map<size_t, size_t> distinctFrequencies;


    std::map<DistinctLeverIndex, Eigen::Matrix<double, 7, 1> > temp_poses;

    std::vector<std::map<size_t, double> > joint_positions(calib_data.joints.size());

    std::set<int> initialized_locations;

    for (size_t i = 0; i < calib_data.calib_frames.size(); i++)
    {
        if (calib_data.calib_frames[i].location_id != -1)
        {
            if (!initialized_locations.count(calib_data.calib_frames[i].location_id))
            {
                auto& location = location_id_to_location[calib_data.calib_frames[i].location_id];
                problem_simple.AddParameterBlock(&location(0), 3);
                problem_simple.AddParameterBlock(&location(3), 4, quaternion_parameterization);

                problem_full.AddParameterBlock(&location(0), 3);
                problem_full.AddParameterBlock(&location(3), 4, quaternion_parameterization2);

                if (calib_data.optional_location_infos.count(
                        calib_data.calib_frames[i].location_id))
                {
                    if (calib_data.optional_location_infos[calib_data.calib_frames[i].location_id]
                            .fixed)
                    {
                        problem_simple.SetParameterBlockConstant(&location(0));
                        problem_simple.SetParameterBlockConstant(&location(3));
                        problem_full.SetParameterBlockConstant(&location(0));
                        problem_full.SetParameterBlockConstant(&location(3));
                    }
                }

                initialized_locations.insert(calib_data.calib_frames[i].location_id);
            }
        }

        const std::vector<int>& jointConfig = calib_data.calib_frames[i].joint_config;

        indexToDistinctJoint[i].resize(calib_data.joints.size());

        for (size_t pj = 0; pj < parent_joints.size(); pj++)
        {
            const size_t j = parent_joints[pj];

#if 0 // one angle variable for each distinct(!) ptu pose
			auto it = distinctJointPositions[j].find(jointConfig[j]);
			if (it == distinctJointPositions[j].end())
			{
				const size_t dj = distinctJointPositions[j].size();
				indexToDistinctJoint[i].push_back(dj);
				distinctJointPositions[j].emplace(jointConfig[j], dj);

				//
				joint_positions[j][dj]
					= jointConfig[j] * 0.051429 / 180.0 * M_PI; //*0.00089779559;
				markerBAProblem.AddParameterBlock(&joint_positions[j][dj], 1);
				markerBAProblem.SetParameterBlockConstant(&joint_positions[j][dj]);

				markerBAProblem_RepError.AddParameterBlock(&joint_positions[j][dj], 1);

				// markerBAProblem_RepError.SetParameterBlockConstant(&joint_positions[j][dj]);
				auto* alphaPrior
					= GaussianPrior1D::Create(joint_positions[j][dj], 0.05 / 180.0 * M_PI);
				markerBAProblem_RepError.AddResidualBlock(
					alphaPrior, nullptr, &joint_positions[j][dj]);
			}
			else
				indexToDistinctJoint[i].push_back(it->second);
#else
            // one angle variable per ptu pose
            indexToDistinctJoint[i][j] = i;
            distinctJointPositions[j].emplace(jointConfig[j], i);

            //
            joint_positions[j][i] = jointConfig[j];

            // std::cout << "Joint val11: " << j << " " << i << " " << joint_positions[j][i] <<
            // std::endl;

            if (calib_data.joints[j].type == "1_dof_joint")
            {

                problem_simple.AddParameterBlock(&joint_positions[j][i], 1);
                problem_simple.SetParameterBlockConstant(&joint_positions[j][i]);

                problem_full.AddParameterBlock(&joint_positions[j][i], 1);

                // Set angular noise
                const double noise_in_ticks = calib_data.joints[j].angular_noise_std_dev
                    / calib_data.joints[j].ticks_to_rad;
                if (noise_in_ticks < 1e-8)
                {
                    problem_full.SetParameterBlockConstant(&joint_positions[j][i]);
                }
                else
                {
                    auto anglePrior
                        = GaussianPrior1D::Create(joint_positions[j][i], noise_in_ticks);
                    problem_full.AddResidualBlock(anglePrior, nullptr, &joint_positions[j][i]);
                }
            }
#endif
        }
        if (descendant_joints.empty())
            continue;

        {
            // one camera pose for each distinct lever config
            std::vector<int>
                leverConfig; //(jointConfig.begin() + upTojointIndex, jointConfig.end());
            // for (size_t pj = 0; pj < descendant_joints.size(); pj++)
            for (size_t descendant_joint_index : descendant_joints)
            {
                // const size_t j = parent_joints[pj];
                leverConfig.push_back(jointConfig[descendant_joint_index]);
            }
            // leverConfig.push_back(calib_data.calib_frames[i].camera_id);

            const auto it = distinctLeverPositions.find(leverConfig);
            if (it == distinctLeverPositions.end())
            {
                const size_t distinct_lever_index = distinctLeverPositions.size();
                indexToDistinctLever[i] = distinct_lever_index;
                distinctLeverPositions.emplace(leverConfig, distinct_lever_index);

                distinctFrequencies[distinct_lever_index] = 1;

                // std::cout << "New Config: ";
                // for (auto i : leverConfig)
                //     std::cout << i << " , ";
                // std::cout << std::endl;

                // std::cout << "Insert CP " << std::endl;

                temp_poses[distinct_lever_index] << 0, 0, 0, 1, 0, 0, 0;
                problem_simple.AddParameterBlock(&temp_poses[distinct_lever_index](0), 3);
                problem_simple.AddParameterBlock(
                    &temp_poses[distinct_lever_index](3), 4, quaternion_parameterization);

                problem_full.AddParameterBlock(&temp_poses[distinct_lever_index](0), 3);
                problem_full.AddParameterBlock(
                    &temp_poses[distinct_lever_index](3), 4, quaternion_parameterization2);
            }
            else
            {
                indexToDistinctLever[i] = it->second;
                // std::cout << "Existing Config: ";
                // for (auto i : leverConfig)
                //     std::cout << i << " , ";
                // std::cout << std::endl;
                distinctFrequencies[it->second]++;
            }
        }
    }

    // std::cout << "num camposes: " << camPoses.size() << std::endl;


    std::vector<std::function<double()> > repErrorFns;
    std::map<int, std::vector<std::function<double()> > > repErrorFnsByCam;


    std::cout << "Building optimization problems..." << std::endl;

    int total_laser_correspondences = 0;

    // curImage=0;
    for (size_t i = 0; i < calib_data.calib_frames.size(); i++)
    {
        // only accept lever groups with at least 2 calibration frames
        if (!descendant_joints.empty())
        {
            if (distinctFrequencies[indexToDistinctLever[i]] < 2)
            {
                // std::cout << "Skipping" << std::endl;
                continue;
            }
        }
        // std::cout << "LeverGroupSize: " << distinctFrequencies[indexToDistinctLever[i]]  <<
        // std::endl;

        // for (const auto& id_to_cam_model : calib_data.cameraModelById)
        for (const auto& sensor_id_to_type : calib_data.sensor_id_to_type)
        {
            const int sensor_id = sensor_id_to_type.first;
            const std::string& sensor_type = sensor_id_to_type.second;
            // const int camera_id = id_to_cam_model.first;

            // Eigen::Matrix<double, 7, 1> cam_to_world = poseInverse(world_to_cam);
            // std::cout << "CamToWorld : " << cam_to_world.transpose() << std::endl;

            TransformationChain chain;


            std::vector<double*> parameter_blocks;

            if (calib_data.calib_frames[i].location_id != -1)
            {
                auto& location = location_id_to_location[calib_data.calib_frames[i].location_id];
                chain.addPose();

                parameter_blocks.push_back(&location(0));
                parameter_blocks.push_back(&location(3));
            }

            const auto path_to_sensor
                = pathFromRootToJoint(calib_data.sensor_id_to_parent_joint[sensor_id]);

            constexpr bool robustify = true; // false; // true;

            // for (size_t pj = 0; pj < parent_joints.size(); pj++)
            for (size_t pj = 0; pj < path_to_sensor.size(); pj++)
            {
                // const size_t j = parent_joints[pj];
                const size_t j = path_to_sensor[pj];
                if (std::find(parent_joints.begin(), parent_joints.end(), j) == parent_joints.end())
                {
                    break;
                }

                const size_t dj = indexToDistinctJoint[i][j];

                if (calib_data.joints[j].type == "1_dof_joint")
                {
                    chain.add1DOFJoint();

                    parameter_blocks.push_back(&jointData[j].parent_to_joint_pose(0));
                    parameter_blocks.push_back(&jointData[j].parent_to_joint_pose(3));
                    parameter_blocks.push_back(&joint_positions[j][dj]);
                    // std::cout << "Joint val11: " << j << " " << dj << " " <<
                    // joint_positions[j][dj] << std::endl;
                    parameter_blocks.push_back(&jointData[j].ticks_to_rad);
                }
                else if (calib_data.joints[j].type == "pose")
                {
                    chain.addPose();

                    parameter_blocks.push_back(&jointData[j].parent_to_joint_pose(0));
                    parameter_blocks.push_back(&jointData[j].parent_to_joint_pose(3));
                }
                // std::cout << "b" << std::endl;
            }
            if (!descendant_joints.empty())
            {
                chain.addPose();
                const size_t dl = indexToDistinctLever[i];
                parameter_blocks.push_back(&temp_poses[dl](0));
                parameter_blocks.push_back(&temp_poses[dl](3));
                //	std::cout << "c" << std::endl;
            }

            if (sensor_type == "camera")
            {
                // sensor is a cam
                const int camera_id = sensor_id;
                const auto& cam_model = calib_data.cameraModelById[camera_id];

                if (reconstructedPoses.count(std::make_pair(i, camera_id)))
                {
                    const Eigen::Matrix<double, 7, 1> world_to_cam
                        = reconstructedPoses[std::make_pair(i, camera_id)];

                    // location_id_to_location[calib_data.calib_frames[i].location_id] =
                    // world_to_cam;
                    // //////// hack

                    auto simpleCostFn = KinematicChainPoseError::Create(world_to_cam, chain);
                    problem_simple.AddResidualBlock(simpleCostFn,
                        robustify ? new ceres::HuberLoss(1.0)
                                  : nullptr, // new ceres::CauchyLoss(3),
                        parameter_blocks);
                }

                // const auto& cam_model = id_to_cam_model.second;
                const auto& camera_observations
                    = calib_data.calib_frames[i].cam_id_to_observations[camera_id];
                const auto& world_points = calib_data.reconstructed_map_points;
                iterateMatches(camera_observations, world_points,
                    [&](int /*point_id*/, const Eigen::Vector2d& cp, const Eigen::Vector3d& wp) {
                        // check origin pose
                        // {
                        //     OpenCVReprojectionError repErr(tagObs.corners[c],
                        //     camModel.distortionCoefficients,camModel.getK());
                        //     repErr.print=true;
                        //     double res[2];
                        //     repErr(&world_to_cam(0), &world_to_cam(3), &tagCorners[c](0), res);
                        //     std::cout << "ERR: " << sqrt(res[0]*res[0]+res[1]*res[1]) <<
                        //     std::endl;
                        // }

                        auto fullCostFn = KinematicChainRepError::Create(
                            cp, wp, cam_model.distortionCoefficients, cam_model.getK(), chain);
                        problem_full.AddResidualBlock(fullCostFn,
                            robustify ? new ceres::HuberLoss(1.0)
                                      : nullptr, // new ceres::CauchyLoss(3),
                            parameter_blocks);

                        repErrorFns.push_back([parameter_blocks, fullCostFn]() -> double {
                            Eigen::Vector2d err;
                            fullCostFn->Evaluate(&parameter_blocks[0], &err(0), nullptr);
                            return err.squaredNorm();
                        });
                        repErrorFnsByCam[camera_id].push_back(
                            [parameter_blocks, fullCostFn]() -> double {
                                Eigen::Vector2d err;
                                fullCostFn->Evaluate(&parameter_blocks[0], &err(0), nullptr);
                                return err.squaredNorm();
                            });
                    });
            }
            else if (sensor_type == "laser_3d")
            {
                //				if (mode == OptimizationMode::SIMPLE_THEN_FULL)
                //					continue;

                const auto& cur_scan
                    = calib_data.calib_frames[i].sensor_id_to_laser_scan_3d[sensor_id];

                const Eigen::Matrix<double, 7, 1> world_to_laser
                    = chain.endEffectorPose(&parameter_blocks[0]);
                const auto laser_to_world = cposeInv<double>(world_to_laser);

                int corresp = 0;
                for (int p = 0; p < cur_scan->points.cols(); p++)
                {
                    const Eigen::Vector3d pt = cur_scan->points.col(p);
                    const Eigen::Vector3d ptw = cposeTransformPoint<double>(laser_to_world, pt);

                    const visual_marker_mapping::ReconstructedTag* min_tag = nullptr;
                    double min_sqr_dist = 9999999999.0;
                    for (const auto& id_to_rec_tag : calib_data.reconstructed_tags)
                    {
                        const visual_marker_mapping::ReconstructedTag& tag = id_to_rec_tag.second;
#if 0
	                    Eigen::Matrix<double, 7, 1> marker_to_world;
	                    marker_to_world.segment<3>(0) = tag.t;
	                    marker_to_world.segment<4>(3) = tag.q;
	                    auto world_to_marker = cposeInv<double>(marker_to_world);
	                    auto mpt = cposeTransformPoint<double>(world_to_marker, ptw);
	
	                    if ((mpt.x() < -0.12) || (mpt.y() < -0.12) || (mpt.x() > 0.12)
	                        || (mpt.y() > 0.12))
	                        continue;
	
	                    if (mpt.z() < -0.2)
	                        continue;
	                    if (mpt.z() > 0.2)
	                        continue;
	
	                    const double dist = mpt.z();
#else
                        const double sqr_dist = (tag.t - ptw).squaredNorm();
                        const double marker_radius
                            = std::max(tag.tagWidth, tag.tagHeight) / 2.0 * sqrt(2.0);
#endif
                        if ((sqr_dist < min_sqr_dist) && (sqr_dist < marker_radius * marker_radius))
                        {
                            min_tag = &tag;
                            min_sqr_dist = sqr_dist;
                        }
                    }
                    if (!min_tag)
                        continue;

                    // dbgout.addPoint(ptw);

                    Eigen::Matrix<double, 7, 1> marker_to_world;
                    marker_to_world.segment<3>(0) = min_tag->t;
                    marker_to_world.segment<4>(3) = min_tag->q;

#if 1
                    auto fullCostFn = MarkerPoint2PlaneError::Create(marker_to_world, pt, chain);
                    problem_full.AddResidualBlock(fullCostFn,
                        robustify ? new ceres::HuberLoss(1.0) : nullptr, parameter_blocks);
#endif


                    //			{
                    //            Eigen::Vector3d err;
                    //            double* parameter_blocks[4] = { &world_to_cam_poses[i](0),
                    //            &world_to_cam_poses[i](3) ,&cam_to_laser_pose(0),
                    //											&cam_to_laser_pose(3)};
                    //            fullCostFn->Evaluate(&parameter_blocks[0], &err(0), nullptr);
                    //            std::cout << "err: " << err.transpose() << std::endl;;
                    //			}

                    corresp++;
                    total_laser_correspondences++;
                }
                // std::cout << "Corresp : "<< corresp << std::endl;
            }
        }
    }
    std::cout << "Building optimization problems...done!" << std::endl;
    std::cout << "Laser correspondence count: " << total_laser_correspondences << std::endl;

    auto computeRMSE = [&repErrorFns]() -> double {
        if (repErrorFns.empty())
            return 0.0;

        double rms = 0;
        for (const auto& errFn : repErrorFns)
        {
            const double sqrError = errFn();
            // if (sqrt(sqrError)>2)
            // std::cout << "RepError: " << sqrt(sqrError) << std::endl;
            rms += sqrError;
        }
        return sqrt(rms / repErrorFns.size());
    };
    auto computeMedianError = [&repErrorFns]() -> double {
        if (repErrorFns.empty())
            return 0.0;

        std::vector<double> errors;
        for (const auto& errFn : repErrorFns)
        {
            const double sqrError = errFn();
            errors.push_back(sqrError);
        }
        std::sort(errors.begin(), errors.end());
        return sqrt(errors[errors.size() / 2]);
    };
    auto computeRMSEByCam = [&repErrorFnsByCam](int cam) -> double {
        if (repErrorFnsByCam[cam].empty())
            return 0.0;

        double rms = 0;
        for (const auto& errFn : repErrorFnsByCam[cam])
        {
            const double sqrError = errFn();
            // std::cout << "RepError: " << sqrt(sqrError) << std::endl;
            rms += sqrError;
        }
        return sqrt(rms / repErrorFnsByCam[cam].size());
    };

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 100;
    options.num_threads = 4;
    // options.num_linear_solver_threads = 4;

#if 0
    for (int j=0;j<jointIndex;j++)
    {
        auto& joint_to_parent_pose = jointData[j].joint_to_parent_pose;
        problem_simple.SetParameterBlockConstant(&joint_to_parent_pose(0));
        problem_simple.SetParameterBlockConstant(&joint_to_parent_pose(3));
    }

    ceres::Solver::Summary summary_pre;
    Solve(options, &problem_simple, &summary_pre);
    std::cout << "Rough Pre Solution: " << summary_pre.termination_type << std::endl;

    for (int j=0;j<jointIndex;j++)
    {
        auto& joint_to_parent_pose = jointData[j].joint_to_parent_pose;
        problem_simple.SetParameterBlockVariable(&joint_to_parent_pose(0));
        problem_simple.SetParameterBlockVariable(&joint_to_parent_pose(3));
    }
#endif


    if ((mode == OptimizationMode::SIMPLE_THEN_FULL) || (mode == OptimizationMode::ONLY_SIMPLE))
    {
        std::cout << "Solving simple optimization problem..." << std::endl;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem_simple, &summary);
        std::cout << "    Simple optimization returned termination type "
                  << summary.termination_type << std::endl;
        // std::cout << summary.FullReport() << std::endl;

        std::cout << "    Simple training reprojection error RMS: " << computeRMSE() << " px"
                  << " Median: " << computeMedianError() << " px" << std::endl;
        std::cout << "Solving simple optimization problem...done!" << std::endl;
    }


    if ((mode == OptimizationMode::SIMPLE_THEN_FULL) || (mode == OptimizationMode::ONLY_FULL))
    {
        std::cout << "Solving full optimization problem..." << std::endl;
        ceres::Solver::Summary summary2;
        ceres::Solve(options, &problem_full, &summary2);
        std::cout << "    Full optimization returned termination type " << summary2.termination_type
                  << std::endl;
        // std::cout << summary2.FullReport() << std::endl;

        std::cout << "    Full training reprojection error RMS: " << computeRMSE() << " px"
                  << " Median: " << computeMedianError() << " px" << std::endl;
        std::cout << "Solving full optimization problem...done!" << std::endl;
    }

    std::cout << std::endl;

    // Print some results
    std::cout << "Resulting Parameters:" << std::endl;

    std::cout << "    Tick2Rad for all joints:\n        ";
    for (size_t pj = 0; pj < parent_joints.size(); pj++)
    {
        const size_t j = parent_joints[pj];
        if (calib_data.joints[j].type == "1_dof_joint")
            std::cout << calib_data.joints[j].name << ":" << jointData[j].ticks_to_rad << ", ";
    }
    std::cout << std::endl;

    std::cout << "    Joint poses:\n";
    for (size_t pj = 0; pj < parent_joints.size(); pj++)
    {
        const size_t j = parent_joints[pj];
        std::cout << "        " << calib_data.joints[j].name << ": "
                  << jointData[j].parent_to_joint_pose.transpose() << std::endl;
    }

    //    std::cout << "Tempoary Poses: " << std::endl;
    //    for (size_t i = 0; i < temp_poses.size(); i++)
    //        std::cout << temp_poses[i].transpose() << std::endl;


    if (!descendant_joints.empty())
    {
        return;
    }
#if 0
    // cameraPose = camPoses[0];
    for (size_t i = 0; i < calib_data.calib_frames.size(); i++)
    {
        const auto& jointConfig = calib_data.calib_frames[i].joint_config;

        Eigen::Matrix<double, 7, 1> root;
        root << 0, 0, 0, 1, 0, 0, 0;

        for (size_t pj = 0; pj < parent_joints.size(); pj++)
        {
            const size_t j = parent_joints[pj];
            root = poseAdd(root, poseInverse(jointData[j].parent_to_joint_pose));

            const double jointAngle = jointConfig[j] * jointData[j].ticks_to_rad;
            // double t = joint_positions[j][indexToDistinctJoint[i][j]];
            Eigen::Matrix<double, 7, 1> tiltRot;
            tiltRot << 0, 0, 0, cos(jointAngle / 2.0), 0, 0, sin(jointAngle / 2.0);
            root = poseAdd(root, tiltRot);

            // auto invRet = poseInverse(root);
            //            DebugVis dbg;
            //            dbg.cam.q = invRet.segment<4>(3);
            //            dbg.cam.t = invRet.segment<3>(0);
            //            dbg.type = 1;
            // debugVis.push_back(dbg);
        }

// std::cout << "NumCamPoses" << camPoses.size() << std::endl;
#if 0
        for (size_t c = 0; c < camPoses.size(); c++)
        {
            auto ret = poseAdd(root, poseInverse(camPoses[c]));
            // std::cout << ret.transpose() << std::endl;
            auto invRet = poseInverse(ret);
            DebugVis dbg;
            dbg.cam.q = invRet.segment<4>(3);
            dbg.cam.t = invRet.segment<3>(0);

            debugVis.push_back(dbg);
        }
#endif
    }

#if 0
    Eigen::Matrix<double, 7, 1> root;
    root << 0, 0, 0, 1, 0, 0, 0;

    for (size_t j = 0; j < jointIndex + 1; j++)
    {
        root = poseAdd(root, poseInverse(jointData[j].joint_to_parent_pose));

        auto invRet = poseInverse(root);
        DebugVis dbg;
        dbg.cam.q = invRet.segment<4>(3);
        dbg.cam.t = invRet.segment<3>(0);
        dbg.type = 1;

        debugVis.push_back(dbg);
    }
#endif
#endif

    if (1)
    {
        // compute rms for forward kinematics
        for (size_t i = 0; i < calib_data.calib_frames.size(); i++)
        {
            const auto& jointConfig = calib_data.calib_frames[i].joint_config;

            for (size_t pj = 0; pj < parent_joints.size(); pj++)
            {
                const size_t j = parent_joints[pj];
                joint_positions[j][i] = jointConfig[j];
            }
        }
        std::cout << "Test Reprojection Error RMS: " << computeRMSE() << " px" << std::endl;

        for (const auto& id_to_cam_model : calib_data.cameraModelById)
        {
            const int camera_id = id_to_cam_model.first;
            std::cout << "Test Reprojection Error RMS for camera " << camera_id << ": "
                      << computeRMSEByCam(camera_id) << " px" << std::endl;
        }
    }
}
//-----------------------------------------------------------------------------
void Calibrator::exportCalibrationResults(const std::string& filePath) const
{
    namespace pt = boost::property_tree;
    pt::ptree root;

    pt::ptree kinematicChainPt;
    for (size_t j = 0; j < calib_data.joints.size(); ++j)
    {
        const auto pose = jointData[j].parent_to_joint_pose;
        const pt::ptree posePt = visual_marker_mapping::matrix2PropertyTreeEigen(pose);

        const auto pose_guess = calib_data.joints[j].parent_to_joint_guess;
        const pt::ptree poseGuessPt = visual_marker_mapping::matrix2PropertyTreeEigen(pose_guess);

        pt::ptree jointDataPt;
        jointDataPt.add_child("parent_to_joint_pose", posePt);
        jointDataPt.add_child("parent_to_joint_pose_guess", poseGuessPt);
        jointDataPt.put("type", calib_data.joints[j].type);
        if (calib_data.joints[j].type == "1_dof_joint")
        {
            jointDataPt.put("ticks_to_rad", jointData[j].ticks_to_rad);
        }
        jointDataPt.put("name", calib_data.joints[j].name);
        jointDataPt.put("parent", calib_data.joints[j].parent);
        jointDataPt.put("fixed", calib_data.joints[j].fixed ? "true" : "false");

        kinematicChainPt.push_back(std::make_pair("", jointDataPt));
    }
    root.add_child("hierarchy", kinematicChainPt);
    pt::ptree locationPt;
    for (const auto& cur_location_id_to_location : location_id_to_location)
    {
        const int location_id = cur_location_id_to_location.first;
        const auto pose = cur_location_id_to_location.second;
        const pt::ptree posePt = visual_marker_mapping::matrix2PropertyTreeEigen(pose);

        pt::ptree curLocationPt;
        curLocationPt.add_child("world_to_location_pose", posePt);
        curLocationPt.put("location_id", location_id);
        if (calib_data.optional_location_infos.count(location_id))
        {
            const auto& loc_info = calib_data.optional_location_infos.find(location_id)->second;
            const auto initial_pose = loc_info.world_to_location_pose_guess;
            const pt::ptree poseGuessPt
                = visual_marker_mapping::matrix2PropertyTreeEigen(initial_pose);
            curLocationPt.add_child("world_to_location_pos_guess", poseGuessPt);

            curLocationPt.put("fixed", loc_info.fixed ? "true" : "false");
        }

        locationPt.push_back(std::make_pair("", curLocationPt));
    }
    root.add_child("locations", locationPt);
    boost::property_tree::write_json(filePath, root);
}
//-----------------------------------------------------------------------------
void Calibrator::calibrate(const std::string& visualization_filename)
{
    for (size_t i = 0; i < calib_data.calib_frames.size(); i++)
    {
        const int location_id = calib_data.calib_frames[i].location_id;
        if (location_id != -1)
        {
            if (!calib_data.optional_location_infos.count(location_id))
            {
                location_id_to_location[location_id] << 0, 0, 0, 1, 0, 0, 0;
            }
            else
            {
                const auto loc_pose
                    = calib_data.optional_location_infos[location_id].world_to_location_pose_guess;
                location_id_to_location[location_id] = loc_pose;
            }
        }

        for (const auto& id_to_cam_model : calib_data.cameraModelById)
        {
            const size_t camera_id = id_to_cam_model.first;
            const auto& cam_model = id_to_cam_model.second;

            Eigen::Quaterniond q;
            Eigen::Vector3d t;
            bool success = computeRelativeCameraPoseFromImg(
                camera_id, i, cam_model.getK(), cam_model.distortionCoefficients, q, t);
            if (!success)
            {
                std::cerr << "Initialization failed" << std::endl;
            }
            else
            {
                const auto cam_pose = cmakePose<double>(t, q);

                //            DebugVis dbg;
                //            dbg.cam.setQuat(q);
                //            dbg.cam.t = t;
                //      if (ptuData.ptuImagePoses[i].cameraId==0)
                //        debugVis.push_back(dbg);

                reconstructedPoses[std::make_pair(i, camera_id)] = cam_pose;
            }
        }
    }

    jointData.resize(calib_data.joints.size());
    for (size_t j = 0; j < jointData.size(); j++)
    {
        jointData[j].parent_to_joint_pose = calib_data.joints[j].parent_to_joint_guess;
        jointData[j].ticks_to_rad = calib_data.joints[j].ticks_to_rad;
        // HACK: Joint positions hier her?
    }

    size_t start_joint = 0;
    std::vector<std::vector<size_t> > joint_to_children(calib_data.joints.size());
    for (size_t j = 0; j < calib_data.joints.size(); j++)
    {
        if (calib_data.joints[j].parent == "base")
        {
            start_joint = j;
            continue;
        }
        size_t parent_j = calib_data.name_to_joint[calib_data.joints[j].parent];
        joint_to_children[parent_j].push_back(j);
    }

    //    std::function<void(size_t)> optimizeJ = [&](size_t j)
    //    {
    //        if (calib_data.joints[j].type == "1_dof_joint")
    //            optimizeUpToJoint(j, false, OptimizationMode::SIMPLE_THEN_FULL);
    //        for (auto cj : joint_to_children[j])
    //            optimizeJ(cj);
    //    };
    //    optimizeJ(start_joint);


    std::set<size_t> frontier;
    frontier.insert(start_joint);

    std::set<size_t> optimization_set;
    // optimization_set.insert(start_joint);
    while (!frontier.empty())
    {
        std::function<void(size_t)> expandToNext1DofJoint = [&](size_t j) {
            if ((calib_data.joints[j].type == "1_dof_joint") && (!optimization_set.count(j)))
            {
                // ptimizeUpToJoint(j, false, OptimizationMode::SIMPLE_THEN_FULL);
                optimization_set.insert(j);
                frontier.insert(j);
                return;
            }
            optimization_set.insert(j);
            for (auto cj : joint_to_children[j])
                expandToNext1DofJoint(cj);
        };
        auto tmp_frontier = frontier;
        frontier.clear();
        for (size_t f : tmp_frontier)
            expandToNext1DofJoint(f);


        std::cout << "Estimating joints: ";
        for (auto o : optimization_set)
            std::cout << calib_data.joints[o].name << ", ";
        std::cout << std::endl;
        //        std::cout << "Frontier Set: " << std::endl;
        //        for (auto o : frontier)
        //            std::cout << calib_data.joints[o].name << ", ";
        //        std::cout << std::endl;


        std::cout << "-------------------------------------------------------------------------"
                  << std::endl;
        // std::cout << "Optim Simple" << std::endl;
        optimizeUpToJoint(optimization_set, OptimizationMode::ONLY_SIMPLE);
        std::cout << "-------------------------------------------------------------------------"
                  << std::endl;
        // std::cout << "Optim Full" << std::endl;
        optimizeUpToJoint(optimization_set, OptimizationMode::SIMPLE_THEN_FULL);
        std::cout << "-------------------------------------------------------------------------"
                  << std::endl;
    }
    // when there are laser sensors, iterate the final optimization a couple of times so that the
    // correspondences can improve
    if (!calib_data.laser_sensor_ids.empty())
    {
        for (int its = 0; its < 20; its++)
        {
			optimizeUpToJoint(optimization_set, OptimizationMode::ONLY_FULL);
            std::cout << "-------------------------------------------------------------------------"
                      << std::endl;
        }
    }

    //////////////////////////////////
    // visualize results
    if (visualization_filename == "")
        return;

    DebugOutput dbg_out(visualization_filename);
    for (size_t i = 0; i < calib_data.calib_frames.size(); i++)
    {
        const std::vector<int>& joint_config = calib_data.calib_frames[i].joint_config;
        std::vector<double> joint_config_d;
        for (auto c : joint_config)
            joint_config_d.push_back(c);

        TransformationChain chain;

        std::vector<double*> parameter_blocks;

        Eigen::Vector3d last_pos(0, 0, 0);

        // Eigen::Matrix<double,7,1> loc;
        // loc << 0,0,0,1,0,0,0;
        if (calib_data.calib_frames[i].location_id != -1)
        {
            auto& location = location_id_to_location[calib_data.calib_frames[i].location_id];
            chain.addPose();

            parameter_blocks.push_back(&location(0));
            parameter_blocks.push_back(&location(3));

            dbg_out.addPose(
                location, "location_" + std::to_string(calib_data.calib_frames[i].location_id));

            // std::cout << "Location: " << location.transpose() << std::endl;

            last_pos = cposeInv(location).segment<3>(0);
        }

        std::map<std::string, Eigen::Matrix<double, 7, 1> > world_to_joint_poses;


        std::function<void(
            const Eigen::Vector3d&, TransformationChain, std::vector<double*>, size_t)> process
            = [&](const Eigen::Vector3d& last_posi, TransformationChain cur_chain,
                std::vector<double*> parameter_blocks, size_t j)
        {
            if (calib_data.joints[j].type == "1_dof_joint")
            {
                cur_chain.add1DOFJoint();

                parameter_blocks.push_back(&jointData[j].parent_to_joint_pose(0));
                parameter_blocks.push_back(&jointData[j].parent_to_joint_pose(3));
                parameter_blocks.push_back(&joint_config_d[j]);
                // std::cout << "Joint val11: " << j << " " << dj << " " <<
                // joint_positions[j][dj] << std::endl;
                parameter_blocks.push_back(&jointData[j].ticks_to_rad);
            }
            else if (calib_data.joints[j].type == "pose")
            {
                cur_chain.addPose();

                parameter_blocks.push_back(&jointData[j].parent_to_joint_pose(0));
                parameter_blocks.push_back(&jointData[j].parent_to_joint_pose(3));
            }

            const auto world_to_pose = cur_chain.endEffectorPose(&parameter_blocks[0]);
            world_to_joint_poses[calib_data.joints[j].name] = world_to_pose;
            // std::cout << "Joint pose of " << calib_data.joints[j].name << "is " <<
            // world_to_pose.transpose() << std::endl;

            dbg_out.addPose(world_to_pose, calib_data.joints[j].name);

            const Eigen::Vector3d cur_pos = cposeInv(world_to_pose).segment<3>(0);

            dbg_out.addLine(last_posi, cur_pos);

            for (size_t cj : joint_to_children[j])
                process(cur_pos, cur_chain, parameter_blocks, cj);
        };
        process(last_pos, chain, parameter_blocks, start_joint);

#if 1 // Show sensor data
        if (i == 0)
        {
            for (const auto& sensor_id_to_type : calib_data.sensor_id_to_type)
            {
                const int sensor_id = sensor_id_to_type.first;
                if (sensor_id_to_type.second == "laser_3d")
                {
                    const std::string& parent_joint
                        = calib_data.sensor_id_to_parent_joint[sensor_id];
                    auto world_to_sensor = world_to_joint_poses[parent_joint];
                    auto sensor_to_world = cposeInv<double>(world_to_sensor);
                    // std::cout << "sensor_to_world: " << sensor_to_world.transpose() << std::endl;

                    const auto& scan
                        = calib_data.calib_frames[i].sensor_id_to_laser_scan_3d[sensor_id];
                    for (int sp = 0; sp < scan->points.cols(); sp++)
                    {
                        const Eigen::Vector3d pw
                            = cposeTransformPoint<double>(sensor_to_world, scan->points.col(sp));
                        if (!std::isnan(pw.x()))
                            dbg_out.addPoint(pw);
                    }
                }
            }
        }
#endif
    }
}
//-----------------------------------------------------------------------------
bool Calibrator::computeRelativeCameraPoseFromImg(size_t camera_id, size_t calibration_frame_id,
    const Eigen::Matrix3d& K, const Eigen::Matrix<double, 5, 1>& distCoefficients,
    Eigen::Quaterniond& q, Eigen::Vector3d& t)
{
    std::vector<Eigen::Vector3d> markerCorners3D;
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > observations2D;
    // find all matches between this image and the reconstructions

    const auto& camera_observations
        = calib_data.calib_frames[calibration_frame_id].cam_id_to_observations[camera_id];
    const auto& world_points = calib_data.reconstructed_map_points;
    iterateMatches(camera_observations, world_points,
        [&](int /*point_id*/, const Eigen::Vector2d& cp, const Eigen::Vector3d& wp) {
            observations2D.push_back(cp);
            markerCorners3D.push_back(wp);
        });

    std::cout << "   Reconstructing camera pose from " << observations2D.size()
              << " 2d/3d correspondences" << std::endl;

    if (observations2D.empty())
        return false;

    Eigen::Matrix3d R;
    // solvePnPEigen(markerCorners3D, observations2D, K, distCoefficients, R, t);
    visual_marker_mapping::solvePnPRansacEigen(
        markerCorners3D, observations2D, K, distCoefficients, R, t);

    q = Eigen::Quaterniond(R);
    return true;
}
//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
