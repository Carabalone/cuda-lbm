#ifndef FLOW_PAST_SQUARE_CYLINDER_FUNCTORS_H
#define FLOW_PAST_SQUARE_CYLINDER_FUNCTORS_H

#include "core/lbm_constants.cuh"
#include "defines.hpp"
#include "functors/includes.cuh"

struct FlowPastSquareCylinderBoundary {
    __host__ __device__
    int operator()(int x, int y, int z) {

        if (x == 0) {
            if (y == 0 || y == NY - 1 || z == 0 || z == NZ - 1)
                return BC_flag::EXTRAPOLATED_CORNER_EDGE;
            else
                return BC_flag::REGULARIZED_INLET_LEFT;
        }

        if (x == NX - 1)
            return BC_flag::ZG_OUTFLOW;

        if (y == 0 || y == NY - 1 || z == 0 || z == NZ - 1)
            return BC_flag::BOUNCE_BACK;

        return BC_flag::FLUID;
    }
};

#endif
