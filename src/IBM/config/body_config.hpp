#pragma once

#include <string>
#include <vector>
#include <map>
#include <optional>
#include "IBM/IBMBody.cuh"

namespace conf {


enum SamplingType {
    DIRECT,
    TARGET_PTS,
    TARGET_SPACING // for selecting r_max ∈ [0.8*Δx, 1.2*Δx];
};
    
struct BodyTransformConfig {
    std::optional<float> scale_overall_size;
    std::optional<float> scale_x;
    std::optional<float> scale_y;
    std::optional<float> scale_z;

    std::optional<float> rotate_x_deg;
    std::optional<float> rotate_y_deg;
    std::optional<float> rotate_z_deg;

    std::optional<float> set_anchor_local_x;
    std::optional<float> set_anchor_local_y;
    std::optional<float> set_anchor_local_z;

    std::optional<float> position_anchor_x;
    std::optional<float> position_anchor_y;
    std::optional<float> position_anchor_z;
};

struct BodySamplingConfig {
    SamplingType type = SamplingType::TARGET_PTS;
    std::optional<int> target_points;
};

struct BodyConfig {
    std::string obj_filepath;
    std::optional<std::string> save_transformed_obj_to; // Optional path to save debug mesh

    BodyTransformConfig transforms;
    BodySamplingConfig sampling;

    BodyConfig() = default;
};

BodyConfig parse_body_config(const std::string& filename);
IBMBody create_body_from_config(const std::string& filename);

} // namespace conf
