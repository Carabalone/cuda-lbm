#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include "third_party/nanoflann/nanoflann.hpp"

struct Sample3;

struct Point3 {
    float x, y, z;

    static std::vector<Point3> from_samples(std::vector <Sample3> samples);

};

struct Sample3 {
    Point3 point;
    float weight;
    int idx;
    bool eliminated;

    bool operator<(const Sample3& other) const {
        return weight < other.weight;
    }

    static std::vector<Sample3> from_points(std::vector<Point3> pts) {
        std::vector<Sample3> samples;
        samples.reserve(pts.size());

        int i = 0;
        for (const auto& pt : pts) {
            samples.push_back({pt, 1.0f, i++, false});
        }

        return samples;
    }
};

inline std::vector<Point3> Point3::from_samples(std::vector <Sample3> samples) {
    std::vector<Point3> points;
    points.reserve(samples.size());

    int i = 0;
    for (const auto& sp : samples) {
        points.push_back(sp.point);
    }

    return points;
}


struct SampleAdaptor {
    const std::vector<Sample3>& samples;

    SampleAdaptor(const std::vector<Sample3>& smp) : samples(smp) {}

    inline size_t kdtree_get_point_count() const {
        return samples.size();
    }

    inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
        switch (dim) {
            case 0:
                return samples[idx].point.x;
            case 1:
                return samples[idx].point.y;
            case 2:
                return samples[idx].point.z;
            default:
                printf("Unknown dim (%d) in kdtree_get_pt\n", dim);
                return 0;
        };
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