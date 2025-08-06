#pragma once

#include <cstdint>
#include <vector>
#include <Eigen/Dense> 

struct InputData {
    // spacepoints
    float sp_x;
    float sp_y;
    float sp_z;
     // cluster info
    float loc_eta_1;
    float loc_phi_1;
    float loc_eta_2;
    float loc_phi_2;

    int64_t athena_id_1;
    int64_t athena_id_2;

    int local_key_1;
    int local_key_2;

    float r;
    float phi;
    float eta;
    float cluster_x_1;
    float cluster_y_1;
    float cluster_z_1;
    float cluster_x_2;
    float cluster_y_2;
    float cluster_z_2;
    float count_1;
    float charge_count_1;
    float localDir0_1;
    float localDir1_1;
    float localDir2_1;
    float lengthDir0_1;
    float lengthDir1_1;
    float lengthDir2_1;
    float glob_eta_1;
    float glob_phi_1;
    float eta_angle_1;
    float phi_angle_1;
    float count_2;
    float charge_count_2;
    float localDir0_2;
    float localDir1_2;
    float localDir2_2;
    float lengthDir0_2;
    float lengthDir1_2;
    float lengthDir2_2;
    float glob_eta_2;
    float glob_phi_2;
    float eta_angle_2;
    float phi_angle_2;
    float cluster_r_1;
    float cluster_phi_1;
    float cluster_eta_1;
    float cluster_r_2;
    float cluster_phi_2;
    float cluster_eta_2;
    float cluster;
};

// struct clusterInfo {
//     std::uint64_t detray_id; 
//     unsigned int local_key;
//     Eigen::Vector3d globalPosition;
//     Eigen::Vector2d localPosition;
//     bool pixel;
// };

struct fittingResult {
    std::vector<float> chi2;
    std::vector<float> ndf;
    std::vector<std::vector<std::array<float, 2>>> local_positions;
    std::vector<std::vector<std::array<float, 2>>> variances;
    std::vector<std::vector<uint64_t>> detray_ids;
    std::vector<std::vector<size_t>> measurement_ids;
    std::vector<std::vector<unsigned int>> measurement_dims;
    std::vector<std::vector<float>> times;
};