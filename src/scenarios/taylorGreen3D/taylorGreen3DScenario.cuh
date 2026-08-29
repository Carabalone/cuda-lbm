#ifndef TAYLOR_GREEN_3D_SCENARIO_H
#define TAYLOR_GREEN_3D_SCENARIO_H

#include "scenarios/scenario.cuh"
#include "taylorGreen3DFunctors.cuh"
#include "postprocessing/kineticEnergy.cuh"

#if defined(LBM_COLLISION_BGK)
using TGV3DCollisionOp = BGK<3>;
#elif defined(LBM_COLLISION_MRT_BASELINE)
using TGV3DCollisionOp = UnoptimizedMRT<3>;
#elif defined(LBM_COLLISION_MRT)
using TGV3DCollisionOp = MRT<3>;
#else
using TGV3DCollisionOp = CM<3, NoAdapter>;
#endif

struct TaylorGreen3DScenario : public ScenarioTrait <
    TaylorGreen3DInit,
    TaylorGreen3DBoundary,
    TaylorGreen3DValidation,
    TGV3DCollisionOp
> {

    // Re=1600
    // N = 64
    // Re = rho0 * u_max * N / vis
    // 1600 = 1.0 * 0.05 * 64 / vis
    // vis = 0.05 * 64 / 1600 = 0.002
    // tau = 0.506 -> really low;
    static constexpr float u_max = 0.05f;
    static constexpr float viscosity = 0.000636619770f;
    static constexpr float tau = viscosity_to_tau(viscosity);
    static constexpr float omega = 1.0f / tau;

    // One common window for new benchmark campaigns. The shorter historical
    // GTX 1050 Ti MRT/AoS protocol is recorded only in its preserved CSV.
    static constexpr int benchmark_timesteps = 350;
    static constexpr int benchmark_warmup = 100;

    static constexpr int timesteps = 8000;
    static constexpr int save_interval = 100;

#if defined(LBM_COLLISION_MRT) || defined(LBM_COLLISION_MRT_BASELINE)
    static constexpr float S[quadratures] = {
        omega, omega, omega, omega, omega,
        omega, omega, omega, omega, omega,
        omega, omega, omega, omega, omega,
        omega, omega, omega, omega, omega,
        omega, omega, omega, omega, omega,
        omega, omega
    };
#else
    static constexpr float S[quadratures] = {0.0f};
#endif

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
        float ke = calculate_kinetic_energy(solver.get_rho(), solver.get_u(), u_max);
        printf("Timestep %d: Kinetic Energy = %.6e\n", solver.timestep, ke);
        
        return -1.0f;
    }
};

#endif // TAYLOR_GREEN_3D_SCENARIO_H
