#ifndef POISEUILLE_CONFIG_H
#define POISEUILLE_CONFIG_H

#include "../functors/initialConditions/poiseuilleInit.cuh"

namespace Config {
    // constexpr float Re = 1000.0f;
    constexpr float u_max = 0.05f;
    constexpr float h_vis = 1.0f / 6.0f;
    constexpr float h_tau   = viscosity_to_tau(h_vis);
    constexpr float h_omega = 1 / h_tau;

    // NY = 50, u_max = 0.067, vis = 1/6
    // Re = 20.

    const PoiseuilleInit init{u_max};

    __host__ __forceinline__
    float poiseuille_analytical(int y) {
        return ((8.0f * h_vis * u_max / (NY*NY)) / (2.0f * h_vis)) * y * (NY-y);
    }
}

#endif //! POISEUILLE_CONFIG_H