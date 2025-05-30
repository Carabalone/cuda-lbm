#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include "IBM/geometry/point.hpp"
#include "nanoflann/nanoflann.hpp"

namespace sampler {

struct Face {
    int v_indices[3];
};

struct MeshData {
    std::vector<geom::Point3D> vertices;
    std::vector<Face> faces;
};

struct Sample3 {
    geom::Point3D point;
    float weight;
    int idx;
    bool eliminated;

    bool operator<(const Sample3& other) const {
        return weight < other.weight;
    }

    static std::vector<Sample3> from_points(std::vector<geom::Point3D> pts) {
        std::vector<Sample3> samples;
        samples.reserve(pts.size());

        int i = 0;
        for (const auto& pt : pts) {
            samples.push_back({pt, 1.0f, i++, false});
        }

        return samples;
    }
};


struct SampleAdaptor {
    const std::vector<Sample3>& samples;

    SampleAdaptor(const std::vector<Sample3>& smp) : samples(smp) {}

    inline size_t kdtree_get_point_count() const {
        return samples.size();
    }

    inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
        return samples[idx].point[dim];
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const {
        return false;
    }
};


typedef nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, SampleAdaptor>,
    SampleAdaptor,
    3
> KDTree;


inline std::vector<geom::Point3D> points_from_samples(const std::vector<sampler::Sample3>& samples) {
    std::vector<geom::Point3D> points;
    points.reserve(samples.size());
    for (const auto& sp : samples) {
        points.push_back(sp.point);
    }
    return points;
}


} // namespace sampler