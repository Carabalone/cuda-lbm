#ifndef TAYLOR_GREEN_3D_SCENARIO_H
#define TAYLOR_GREEN_3D_SCENARIO_H

#include "scenarios/scenario.cuh"
#include "taylorGreen3DFunctors.cuh"
#include "postprocessing/kineticEnergy.cuh"

struct TaylorGreen3DScenario : public ScenarioTrait <
    TaylorGreen3DInit,
    TaylorGreen3DBoundary,
    TaylorGreen3DValidation,
    BGK<3>
> {

    // Re=1600
    // N = 64
    // Re = rho0 * u_max * N / vis
    // 1600 = 1.0 * 0.05 * 64 / vis
    // vis = 0.05 * 64 / 1600 = 0.002
    // tau = 0.506 -> really low;
    static constexpr float u_max = 0.05f;
    static constexpr float viscosity = 0.002f;
    static constexpr float tau = viscosity_to_tau(viscosity);
    static constexpr float omega = 1.0f / tau;

    static constexpr float S[quadratures] = {0.0f};
    
    static const char* name() { return "TaylorGreen3D"; }
    
    static InitType init() { 
        return InitType(u_max); 
    }

    static BoundaryType boundary() {
        return BoundaryType();
    }
    
    static ValidationType validation() {
        return ValidationType(u_max, viscosity, t);
    }
    
    template <typename LBMSolver>
    static float compute_error(LBMSolver& solver) {
        // we are computing in GPU arrays so no problems with data freshness here.
        // if (solver.update_ts < solver.timestep)
        //     solver.update_macroscopics();

        // auto v = validation();

        float ke = calculate_kinetic_energy(solver.get_rho(), solver.get_u());
        printf("Timestep %d: Kinetic Energy = %.6e\n", solver.timestep, ke);
        
        return -1.0f;
    }
};

#endif // TAYLOR_GREEN_3D_SCENARIO_H