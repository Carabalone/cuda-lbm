#pragma once

#include <cmath>
#include <array>
#include <cstddef>
#include <vector>

namespace sampler {

struct Sample3;

}

namespace geom {

template <int dim>
struct Point {
    std::array<float, dim> coords;

    Point() = default;
    
    Point(float x, float y) : coords{x, y} {
        static_assert(dim == 2, "tried to use 2D-only constructor with dim!=2");
    }
    
    Point(float x, float y, float z) : coords{x, y, z} {
        static_assert(dim == 3, "tried to use 3D-only constructor with dim!=3");
    }
    
    float& operator[](size_t i) { return coords[i]; }
    const float& operator[](size_t i) const { return coords[i]; }
    
    float x() const { return coords[0]; }
    float y() const { return coords[1]; }
    float z() const { 
        static_assert(dim >= 3, "z can only be used in Point<3>\n");
        return coords[2]; 
    }

    float* get_data_ptr() {
        return coords.data();
    }
    
};

using Point2D = Point<2>;
using Point3D = Point<3>;

} // namespace geom;