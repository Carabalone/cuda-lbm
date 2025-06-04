#pragma once
#include <string>
#include "IBM/mesh/mesh.hpp"

namespace mesh {

MeshData load_obj(const std::string& filename);

void save_obj(const MeshData& mesh, const std::string& filename);

} // namespace mesh
