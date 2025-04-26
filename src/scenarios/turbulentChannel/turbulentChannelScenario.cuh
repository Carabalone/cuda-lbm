#ifndef TURBULENT_CHANNEL_SCENARIO_H
#define TURBULENT_CHANNEL_SCENARIO_H

#include "scenarios/scenario.cuh"
#include "turbulentChannelFunctors.cuh"

struct TurbulentChannelScenario : public ScenarioTrait <
    TurbulentChannelInit,
    TurbulentChannelBoundary,
    TurbulentChannelValidation,
    MRT<3>
> {
    static constexpr float Re = 3300.0f;
    static constexpr float Re_tau = 180.0f;
    static constexpr int N_half = NY / 2;
    
    // static constexpr float u_max = 0.1f;
    static constexpr float u_tau = 0.00579f; // Mach number of 0.01
    static constexpr float viscosity = u_tau * N_half / Re_tau;
    static constexpr float u_max = viscosity * Re / N_half;
    // static constexpr float u_tau = viscosity * Re_tau / N_half;
    static constexpr float tau = viscosity_to_tau(viscosity);
    static constexpr float omega = 1.0f / tau;

    static constexpr float perturbation = 0.1f;

    // Suga et. al (2015)
    // A D3Q27 multiple-relaxation-time lattice Boltzmann method for turbulent flows
    static constexpr float S[quadratures] = {
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        omega,
        omega,
        omega,
        omega,
        omega,
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        1.0f
    };
    
    // // Lallemand and Luo (2000) magic parameters.
    // static constexpr float S[quadratures] = {
    //     0.0f,  // density (conserved)
    //     1.63f, // bulk viscosity related - controls compressibility
    //     1.14f, // energy flux tensor
    //     0.0f,  // momentum-x (conserved)
    //     1.92f, // energy square moment - affects stability in high Reynolds number flows
    //     0.0f,  // momentum-y (conserved)
    //     1.98f, // third-order moment - affects numerical stability near boundaries
    //     omega, // kinematic viscosity (shear viscosity)
    //     omega  // kinematic viscosity (shear viscosity)
    // };
    // static constexpr float S[quadratures] = {0.0f};
    
    static const char* name() { return "TurbulentChannel"; }
    
    static InitType init() { 
        return InitType(u_max, u_tau, perturbation); 
    }

    static BoundaryType boundary() {
        return BoundaryType();
    }
    
    static ValidationType validation() {
        return ValidationType();
    }
    
    template <typename LBMSolver>
    static float compute_error(LBMSolver& solver) {
        if (solver.update_ts < solver.timestep)
            solver.update_macroscopics();

        return -1.0f;
    }
};

#endif // !TURBULENT_CHANNEL_SCENARIO_H