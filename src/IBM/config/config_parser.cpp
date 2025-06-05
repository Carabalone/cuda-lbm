#include "IBM/config/body_config.hpp"
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iostream> 
#include <algorithm>
#include "IBM/mesh/mesh_transformer.hpp"
#include <cmath>
#include "defines.hpp"
#include "core/math_constants.cuh"

namespace conf {

static std::string trim_str(const std::string& str) {

    const std::string whitespace = " \t\n\r\f\v";
    size_t start = str.find_first_not_of(whitespace);
    if (start == std::string::npos) return "";
    size_t end = str.find_last_not_of(whitespace);

    return str.substr(start, (end - start + 1));
}

static std::string lower_str(std::string s) {

    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return s;
}


BodyConfig parse_body_config(const std::string& config_filepath) {
    std::ifstream config_file(config_filepath);
    if (!config_file.is_open()) {
        std::cerr << (" [ERROR] Failed to open configuration file: " + config_filepath) << std::endl;
        exit(EXIT_FAILURE);
    }

    BodyConfig config;
    std::string current_line;
    std::string current_section;
    int line_number = 0;

    config.sampling.type = SamplingType::TARGET_PTS;

    while (std::getline(config_file, current_line)) {
        line_number++;
        current_line = trim_str(current_line);

        if (current_line.empty() || current_line[0] == '#') {
            continue;
        }

        if (current_line[0] == '[' && current_line.back() == ']') {
            current_section = trim_str(current_line.substr(1, current_line.length() - 2));
            current_section = lower_str(current_section);
            continue;
        }

        size_t equals_pos = current_line.find('=');
        if (equals_pos == std::string::npos) {
            std::cerr << "[Warning] ConfigParser: Malformed line " << line_number << " (no '='): " << current_line << std::endl;
            continue;
        }

        std::string key = trim_str(current_line.substr(0, equals_pos));
        std::string value_str = trim_str(current_line.substr(equals_pos + 1));
        key = lower_str(key);

        if (value_str.empty()) {
            std::cerr << "[Warning] ConfigParser: Empty value for key '" << key << "' on line " << line_number << std::endl;
            continue;
        }
        
        if (current_section == "general") {
            if (key == "obj_filepath") {
                config.obj_filepath = value_str;
            } else if (key == "save_transformed_obj_to") {
                config.save_transformed_obj_to = value_str;
            }
        }
        else if (current_section == "transformations") {
            if (key == "scale_overall_size") config.transforms.scale_overall_size = std::stof(value_str);
            else if (key == "scale_x") config.transforms.scale_x = std::stof(value_str);
            else if (key == "scale_y") config.transforms.scale_y = std::stof(value_str);
            else if (key == "scale_z") config.transforms.scale_z = std::stof(value_str);
            else if (key == "rotate_x_deg") config.transforms.rotate_x_deg = std::stof(value_str);
            else if (key == "rotate_y_deg") config.transforms.rotate_y_deg = std::stof(value_str);
            else if (key == "rotate_z_deg") config.transforms.rotate_z_deg = std::stof(value_str);
            else if (key == "set_anchor_local_x") config.transforms.set_anchor_local_x = std::stof(value_str);
            else if (key == "set_anchor_local_y") config.transforms.set_anchor_local_y = std::stof(value_str);
            else if (key == "set_anchor_local_z") config.transforms.set_anchor_local_z = std::stof(value_str);
            else if (key == "position_anchor_x") config.transforms.position_anchor_x = std::stof(value_str);
            else if (key == "position_anchor_y") config.transforms.position_anchor_y = std::stof(value_str);
            else if (key == "position_anchor_z") config.transforms.position_anchor_z = std::stof(value_str);
        }
        else if (current_section == "sampling") {
            if (key == "type") {
                std::string type_val = lower_str(value_str);

                if (type_val == "direct") config.sampling.type = SamplingType::DIRECT;

                else if (type_val == "target_points") config.sampling.type = SamplingType::TARGET_PTS;

                else std::cerr << "[Warning] ConfigParser: Unknown sampling type '" << value_str << "' on line " << line_number << std::endl;
            }
            else if (key == "target_points") {
                config.sampling.target_points = std::stoi(value_str);
            }
        } else {
            std::cerr << "[Warning] ConfigParser: Unknown section '" << current_section << "' on line " << line_number << std::endl;
        }
    }
    config_file.close();

    if (config.obj_filepath.empty()) {
        std::cerr << ("[ERROR]'obj_filepath' not specified in [General] section of " + config_filepath) << std::endl;
        exit(EXIT_FAILURE);
    }
    if (config.sampling.type == SamplingType::TARGET_PTS && !config.sampling.target_points.has_value()) {
        std::cerr << ("[ERROR] Sampling type is TARGET_POINTS but 'target_points' not specified in [Sampling] section of " + config_filepath) << std::endl;
    }
    if (config.sampling.target_points.has_value() && config.sampling.target_points.value() <= 0 && config.sampling.type == SamplingType::TARGET_PTS) {
        std::cout << ("[ERROR] 'target_points' must be positive if specified. File: " + config_filepath) << std::endl;
    }


    return config;
}

IBMBody create_body_from_config(const std::string& config_filepath) {

    BodyConfig config;
    config = parse_body_config(config_filepath);

    mesh::MeshTransformer transformer(config.obj_filepath);
    std::cout << "[IBM Create] Loaded OBJ: " << config.obj_filepath << std::endl;

    if (config.transforms.scale_overall_size.has_value()) {
        transformer.scale_to_overall_size(config.transforms.scale_overall_size.value());
    } else if (config.transforms.scale_x.has_value() || config.transforms.scale_y.has_value() || config.transforms.scale_z.has_value()) {
        transformer.scale(config.transforms.scale_x.value_or(1.0f),
                          config.transforms.scale_y.value_or(1.0f),
                          config.transforms.scale_z.value_or(1.0f));
    }

    if (config.transforms.rotate_y_deg.has_value()) transformer.rotate_y(deg2rad(config.transforms.rotate_y_deg.value()));
    if (config.transforms.rotate_x_deg.has_value()) transformer.rotate_x(deg2rad(config.transforms.rotate_x_deg.value()));
    if (config.transforms.rotate_z_deg.has_value()) transformer.rotate_z(deg2rad(config.transforms.rotate_z_deg.value()));

    if (config.transforms.set_anchor_local_x.has_value() &&
        config.transforms.set_anchor_local_y.has_value() &&
        config.transforms.set_anchor_local_z.has_value()) {
        transformer.set_anchor(config.transforms.set_anchor_local_x.value(),
                               config.transforms.set_anchor_local_y.value(),
                               config.transforms.set_anchor_local_z.value());
    }

    if (config.transforms.position_anchor_x.has_value() &&
        config.transforms.position_anchor_y.has_value() &&
        config.transforms.position_anchor_z.has_value()) {
        transformer.move_anchor_to_world(config.transforms.position_anchor_x.value(),
                                         config.transforms.position_anchor_y.value(),
                                         config.transforms.position_anchor_z.value());
    }

    if (config.save_transformed_obj_to.has_value() && !config.save_transformed_obj_to.value().empty()) {
        transformer.collect_file(config.save_transformed_obj_to.value());
    }

    if (!transformer.fits_in_domain(NX, NY, NZ)) {
        std::cerr << "[ERROR] Body from \"" << config_filepath
                  << "\" may not fit domain after transformations." << std::endl;
        exit(EXIT_FAILURE);
    }

    IBMBody body;
    if (config.sampling.type == SamplingType::DIRECT) {
        body = transformer.collect_ibm_body();
    } else if (config.sampling.type == SamplingType::TARGET_PTS) {
        body = transformer.collect_ibm_body(config.sampling.target_points.value());
    } else {
        std::cerr << "[IBM ERROR] Unknown sampling type from config. Should have been caught by parser." << std::endl;
        exit(EXIT_FAILURE);
    }

    if (body.num_points > 0) {
        std::cout << "[IBM Create] Successfully created IBMBody with " << body.num_points << " points." << std::endl;

        mesh::AABB aabb = transformer.get_aabb();
        std::cout << "[IBM Create] Final Transformed Mesh AABB (before sampling): "
                  << "Min(" << aabb.min_ext.x() << ", "
                  << aabb.min_ext.y() << ", "
                  << aabb.min_ext.z() << ") "
                  << "Max(" << aabb.max_ext.x() << ", "
                  << aabb.max_ext.y() << ", "
                  << aabb.max_ext.z() << ")" << std::endl;
        geom::Point3D size = aabb.get_size();
        std::cout << "[IBM Create] Final Transformed Mesh Size: ("
                  << size.x() << ", " << size.y() << ", " << size.z() << ")" << std::endl;
    } else {
        std::cerr << "[ERROR] Resulting IBMBody has 0 points for config: " << config_filepath << std::endl;
        exit(EXIT_FAILURE);
    }
    
    return body;
}

} // namespace conf
