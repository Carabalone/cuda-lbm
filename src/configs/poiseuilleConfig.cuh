
#ifndef POISEUILLE_CONFIG_H
#define POISEUILLE_CONFIG_H

#include "../functors/initialConditions/poiseuilleInit.cuh"

namespace Config {
    // constexpr float Re = 1000.0f;
    constexpr float u_max = 0.1f;
    constexpr float h_vis = 1.0f / 6.0f;
    constexpr float h_tau   = viscosity_to_tau(h_vis);
    constexpr float h_omega = 1 / h_tau;

    const PoiseuilleInit init{u_max};
}

#endif //! POISEUILLE_CONFIG_H