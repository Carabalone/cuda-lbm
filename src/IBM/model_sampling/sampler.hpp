#pragma once

#include "IBM/model_sampling/points.hpp"
#include <queue>

namespace sampler {

    float calculate_mesh_volume(const std::vector<Point3>& vertices,
                                const std::vector<Face>& faces);
    float calculate_r_max_3d(float domain_volume, int target_samples);

    struct Sampler {

        int target_samples;
        float r_max;
        std::vector<Sample3>& samples;
        const int alpha = 8;

        SampleAdaptor adaptor;
        KDTree tree;
        std::priority_queue<Sample3> heap;
        
        Sampler(int target, float r_max, std::vector<Sample3>& samples) : target_samples(target),
                r_max(r_max), samples(samples), adaptor(samples),
                tree(3, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10)) {

            printf("Building tree...\n");
            tree.buildIndex();
            printf("Tree built\n");
        }

        static float euclidean_dist(Point3& p1, Point3& p2) {
            float dx = p2.x - p1.x;
            float dy = p2.y - p1.y;
            float dz = p2.z - p1.z;

            return sqrtf(dx * dx + dy * dy + dz * dz);
        }

        float calculate_wij(Point3& p1, Point3& p2) {
            float d = euclidean_dist(p1, p2);
            if (d >= 2.0f * r_max)
                return 0.0f;

            return std::pow(1.0f - d / (2.0f * r_max), alpha);
        }

        void assign_weights() {
            int i = 0;
            for (auto& sample : samples) {
                printf("[WEIGHTS]: %.2f%\r", (static_cast<float>(i) / samples.size()) * 100.0f);
                sample.weight = 0.0f;

                Point3& anchor_point = sample.point;

                float two_r_max_sq = 4.0f * r_max * r_max;
                std::vector<nanoflann::ResultItem<uint32_t, float>> results;
                tree.radiusSearch(&anchor_point.x, two_r_max_sq, results, nanoflann::SearchParameters());

                for (auto& res : results) {
                    uint32_t neigh_idx = res.first;

                    if (neigh_idx >= samples.size())
                        continue;

                    sample.weight += sample.idx == samples[neigh_idx].idx ?
                                        0.0f :
                                        calculate_wij(anchor_point, samples[neigh_idx].point);
                    
                }

                i++;
            }
            printf("\n");
        }

        std::priority_queue<Sample3> build_heap() {
            std::priority_queue<Sample3> sample_heap;
            for (const auto& s : samples) {
                if (!s.eliminated) {
                    sample_heap.push(s);
                }
            }
            return sample_heap;
        }

        std::vector<Point3> eliminate_samples() {
            int num_samples = samples.size();
            int removed_samples = 0;

            if (target_samples >= num_samples) {
                printf("Cannot perform sample elimination (target >= num_samples)\n");

                return Point3::from_samples(samples);
            }

            printf("Assigning weights...\n");
            assign_weights();
            printf("Building heap\n");
            heap = build_heap();
            printf("Heap built.\n");

            while (num_samples - removed_samples > target_samples) {
                if (heap.size() <= 0) {
                    printf("Heap is empty");
                    exit(1);
                }

                Sample3 s = heap.top();
                heap.pop();
                // heap contains values, true samples are in sample& samples array.
                Sample3& s_ref = samples[s.idx];

                if (s_ref.eliminated)
                    continue;
                // check if value is stale or not
                if (std::fabs(s_ref.weight - s.weight) > 1e-6f)
                    continue;

                Sample3& s_top = s_ref; // just renaming so algo is clearer to read
                s_top.eliminated = true;
                removed_samples++;

                float search_radius_sq = (2.0f * r_max) * (2.0f * r_max); //2r_max from paper, but nanoflann expects r² instead of r
                std::vector<nanoflann::ResultItem<uint32_t, float>> results;
                tree.radiusSearch(
                    &s_top.point.x, search_radius_sq, results, nanoflann::SearchParameters()
                );

                // if (/*(num_samples - removed_samples) % 100 == 0 &&*/ removed_samples > 0) {
                    printf("%.2f% | Removed %d samples\r",  ((static_cast<float>(removed_samples)) / (num_samples - target_samples)) * 100.0f , removed_samples);
                // }

                for (auto& item : results) {
                    Sample3& s_neigh = samples[item.first];
                    if (s_neigh.eliminated || s_neigh.idx == s_top.idx)
                        continue;

                    s_neigh.weight -= calculate_wij(s_neigh.point, s_top.point);

                    heap.push(s_neigh);
                }
            }

            printf("\n");

            std::vector<Point3> final_points;
            final_points.reserve(num_samples - removed_samples);
            for (auto& s_obj : samples) {
                if (!s_obj.eliminated) {
                    final_points.push_back(s_obj.point);
                }
            }
            return final_points;
        }
    };
}
