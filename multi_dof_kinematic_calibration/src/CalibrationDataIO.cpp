#include "multi_dof_kinematic_calibration/CalibrationDataIO.h"
#include "visual_marker_mapping/DetectionIO.h"

#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <Eigen/Core>

#include "visual_marker_mapping/CameraUtilities.h"
#include "visual_marker_mapping/ReconstructionIO.h"

namespace multi_dof_kinematic_calibration
{
//----------------------------------------------------------------------------
CalibrationData::CalibrationData(const std::string& filePath)
{
    namespace pt = boost::property_tree;
    pt::ptree rootNode;
    pt::read_json(filePath, rootNode);

    // Read Reconstructions
    {
        boost::filesystem::path reconstruction_filename
            = rootNode.get<std::string>("world_reference");
        if (reconstruction_filename.is_relative())
            reconstruction_filename
                = boost::filesystem::path(filePath).parent_path() / reconstruction_filename;
        visual_marker_mapping::CameraModel cam_model;
        std::map<int, visual_marker_mapping::Camera> reconstructed_cameras;
        std::map<int, visual_marker_mapping::ReconstructedTag> reconstructed_tags;
        visual_marker_mapping::parseReconstructions(
            reconstruction_filename.string(), reconstructed_tags, reconstructed_cameras, cam_model);
        std::cout << "Read reconstructions!" << std::endl;

        reconstructed_map_points = visual_marker_mapping::flattenReconstruction(reconstructed_tags);
    }

    for (const auto& jointNode : rootNode.get_child("kinematic_chain"))
    {
        JointInfo inf;
        inf.name = jointNode.second.get<std::string>("name");
        inf.ticks_to_rad = jointNode.second.get<double>("ticks_to_rad");
        inf.angular_noise_std_dev = jointNode.second.get<double>("angular_noise_std_dev");
        joints.emplace_back(std::move(inf));
    }

    std::map<int, visual_marker_mapping::DetectionResult> detectionResultsByCamId;
    for (const auto& cameraNode : rootNode.get_child("cameras"))
    {
        const int camera_id = cameraNode.second.get<int>("camera_id");
        boost::filesystem::path camera_path = cameraNode.second.get<std::string>("camera_path");

        if (camera_path.is_relative())
            camera_path = boost::filesystem::path(filePath).parent_path() / camera_path;

        // Read cam intrinsics

        const std::string cam_intrin_filename = (camera_path / "camera_intrinsics.json").string();

        std::cout << "Trying to load camera intrinsics for camera " << camera_id << " from "
                  << cam_intrin_filename << " ..." << std::endl;

        boost::property_tree::ptree cameraTree;
        boost::property_tree::json_parser::read_json(cam_intrin_filename, cameraTree);
        cameraModelById.emplace(
            camera_id, visual_marker_mapping::propertyTreeToCameraModel(cameraTree));

        // Read marker detections

        const std::string marker_det_filename = (camera_path / "marker_detections.json").string();

        std::cout << "Trying to load marker detections for camera " << camera_id << " from "
                  << marker_det_filename << " ..." << std::endl;
        detectionResultsByCamId.emplace(
            camera_id, visual_marker_mapping::readDetectionResult(marker_det_filename));
        std::cout << "Read marker detections for camera " << camera_id << "!" << std::endl;
    }

    //	std::ofstream conv("conv.txt");
    for (const auto& ptuPoseNode : rootNode.get_child("calibration_frames"))
    {
        CalibrationFrame ptuInfo;

        for (const auto& id_to_cam_model : cameraModelById)
        {
            const int camera_id = id_to_cam_model.first;
            char buffer[256];
            sprintf(buffer, "camera_image_path_%d", camera_id);
            boost::filesystem::path image_path = ptuPoseNode.second.get<std::string>(buffer);
            if (image_path.is_relative())
                image_path = boost::filesystem::path(filePath).parent_path() / image_path;

            // find marker observations for this calibration frame
            int detectedImageId = -1;
            for (size_t j = 0; j < detectionResultsByCamId[camera_id].images.size(); j++)
            {
                // this is a hack and should really be ==...
                if (image_path.filename() == detectionResultsByCamId[camera_id].images[j].filename)
                {
                    detectedImageId = static_cast<int>(j);
                    break;
                }
            }
            for (const auto& tagObs : detectionResultsByCamId[camera_id].tagObservations)
            {
                if (tagObs.imageId != detectedImageId)
                    continue;
				
                for (size_t c = 0; c < 4; c++)
                {
                    const int point_id = tagObs.tagId * 4 + c;
                    ptuInfo.cam_id_to_observations[camera_id][point_id] = tagObs.corners[c];
                }
            }
        }

        // ptuInfo.image_path = image_path.string();
        // ptuInfo.camera_id = ptuPoseNode.second.get<int>("camera_id");

        for (size_t j = 0; j < joints.size(); j++)
        {
            char buffer[256];
            sprintf(buffer, "joint_ticks_%d", j);
            const int ticks = ptuPoseNode.second.get<int>(buffer);
            ptuInfo.joint_config.push_back(ticks);
        }

//		if (ptuInfo.camera_id==0)
//		{
//			conv << "{\n";
//			conv << "\"location_id\": 0," << std::endl;
//			conv << "\"camera_image_path_0\": \"" << origim.string() << "\"," << std::endl;
//			std::string tmp=origim.string();
//			tmp[3]='1';
//			conv << "\"camera_image_path_1\": \"" << tmp << "\"," << std::endl;
//			conv << "\"joint_ticks_0\": \"" << ptuPoseNode.second.get<int>("joint_0.ticks") << "\","
//<<
// std::endl;
//			conv << "\"joint_ticks_1\": \"" << ptuPoseNode.second.get<int>("joint_1.ticks") << "\","
//<<
// std::endl;
//			conv << "},\n";
//		}

// ptuInfo.cameraIdToObservations

#if 0
        // find marker observations for this calibration frame
        int detectedImageId = -1;
        for (size_t j = 0; j < detectionResultsByCamId[ptuInfo.camera_id].images.size(); j++)
        {
            // this is a hack and should really be ==...
            if (image_path.filename()
                == detectionResultsByCamId[ptuInfo.camera_id].images[j].filename)
            {
                detectedImageId = static_cast<int>(j);
                break;
            }
        }
        for (const auto& tagObs : detectionResultsByCamId[ptuInfo.camera_id].tagObservations)
        {
            if (tagObs.imageId != detectedImageId)
                continue;
            ptuInfo.marker_observations.push_back(tagObs);
            ptuInfo.marker_observations.back().imageId = -1;
        }
#endif

        calib_frames.emplace_back(std::move(ptuInfo));
    }
}
//----------------------------------------------------------------------------
}