#ifndef ADAPTERS_H
#define ADAPTERS_H
#include "core/lbm_constants.cuh"

// currently only for 2D
// will extend to 3D later.
struct AdapterBase {
    static constexpr int i_star = 5;
    
    __device__ __forceinline__
    static bool is_higher_order(int moment_idx) {
        return moment_idx > i_star;
    }
};

struct NoAdapter : public AdapterBase {
    __device__ __forceinline__
    static float compute_higher_order_relaxation(float rho, float j_mag, float pi_mag,
                                              MomentInfo avg_mag) {
        return 1.0f;
    }
};

struct OptimalAdapter : public AdapterBase {
    __device__ __forceinline__
    static float compute_higher_order_relaxation(float rho, float j_mag, float pi_mag,
                                              MomentInfo avg_mag) {
        // W. Li, Y. Chen, M. Desbrun, C. Zheng, and X. Liu, “Fast and Scalable Turbulent Flow Simulation
        // with Two-Way Coupling,” ACM Trans. Graph., vol. 39, no. 4, p. 47, 2020.
        // θ* = (0.0003, -0.00775, 0.00016, 0.0087)
        float s_p[4] = {
            rho    / avg_mag.rho_avg_norm,
            j_mag  / avg_mag.j_avg_norm,
            pi_mag / avg_mag.pi_avg_norm,
            1.0f  // affine term
        };
        
        float theta[4] = {0.0003f, -0.00775f, 0.00016f, 0.0087f};
        float tau_star = 0.0f;
        
        for (int i = 0; i < sizeof(theta)/sizeof(float); i++) {
            tau_star += theta[i] * s_p[i];
        }
        
        // Look, in the paper Fast and Scalable Turbulent Flow Simulation with Two-Way Coupling
        // the authors just say: "if a clear local minimum is found, we use that time
        // as our new 𝜏∗ at this node; if no local minimum is found (in the rare
        // cases discussed above), we simply keep the previous local time used
        // at the previous time step."
        // however, finding the "rare cases" with the regression model is a little bit more complicated.
        // since we use regression, there is no way to know about the minima.
        // This means that for inflow boundaries, where the velocity magnitude is likely very different
        // from the rest of the solver, tau* explodes.
        // Empirically, the researches found that tau_star=0.005f is a good starting value.
        //
        // lets consider adapted_v from the empirical ACM:
        // adapted_v = (-4 * u_mag / max_u_mag + 5) * art_vis.
        // where art_vis are artificial viscosities with max = 0.001 and min = 0.0005 
        // for u_mag = 0.0f and max_u_mag != 0
        // the values we van have for art_vis are {min=0.0025, max=0.005},
        // for u_mag = max_u_mag, the values we can have for
        // art_vis are {min=0.0005, max=0.001}, so for all values,
        // art_vis ∈ [0.0005, 0.0025].
        // assuming that they use these viscosities in the normal tau calculation:
        // tau = 1/(3v + 0.5), the values we can have are:
        // tau = {v_min=}
        // tau ∈ []
        // I save this here but it makes not much sense.
        // as a temporary solution I am just going to do one thing:
        // if tau_star is negatve -> put at 0.005f (recommended value) 
        // clamp tau_star from 0.0f to 1.5f and monitor the values for now
        
        float tau_star_old = tau_star;
        tau_star = tau_star > 0.0f ? tau_star : 0.5f;
        tau_star = fminf(tau_star, 1.5f);

        if (fabsf(tau_star_old) > 0.5){
            printf("{\n\trho: %.4f\n" "\ts_p[0]: %.4f\n" "\ts_p[1]: %.4f\n"
                   "\ts_p[2]: %.8f\n" "\ts_p[3]: %.4f\n" "\ttau_star: %.4f\n"
                   "\tpi_mag/pi_mag_norm: %.6f/%.6f\n\tnew_tau_star: %.8f\n}\n",
                    rho, s_p[0], s_p[1], s_p[2], s_p[3], tau_star_old,
                    pi_mag, avg_mag.pi_avg_norm, tau_star);
        }

        return 1.0f / (tau_star + 0.5f);
    }
};

#endif // ! ADAPTERS_H