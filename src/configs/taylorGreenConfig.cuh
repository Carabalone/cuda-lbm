#ifndef TAYLOR_GREEN_CONFIG_H
#define TAYLOR_GREEN_CONFIG_H

#include "../functors/initialConditions/taylorGreenInit.cuh"

namespace Config {
    // constexpr float Re = 1000.0f;
    constexpr float u_max = 0.04f;
    constexpr float h_vis = 1.0f / 6.0f;
    constexpr float h_tau   = viscosity_to_tau(h_vis);
    constexpr float h_omega = 1 / h_tau;

    const TaylorGreenInit init{h_vis};
    // constexpr float h_vis = u_max * NY / Re;
}

#endif //! TAYLOR_GREEN_CONFIG_H