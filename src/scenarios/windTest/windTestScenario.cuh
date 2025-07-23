#ifndef WIND_TUNNEL_SCENARIO_H
#define WIND_TUNNEL_SCENARIO_H

#include "scenarios/scenario.cuh"
#include "functors/initialConditions/defaultInit.cuh"
#include "IBM/config/body_config.hpp"

/* ------------------------------------------------------------------ */
/*  Boundary functor                                                    */
/* ------------------------------------------------------------------ */
struct WindTunnelBoundary
{
    __host__ __device__
    int operator()(int x, int y, int z) const
    {
        const int edges = (x==0) + (x==NX-1) +
                          (y==0) + (y==NY-1) +
                          (z==0) + (z==NZ-1);

        // if (edges >= 2) return BC_flag::EXTRAPOLATED_CORNER_EDGE;

        if (x == 0)       return BC_flag::GUO_VELOCITY_INLET;
        // if (x == NX-1)    return BC_flag::PRESSURE_OUTLET;

        // if (x == 0)       return BC_flag::REGULARIZED_INLET_LEFT;
        if (x == NX-1)    return BC_flag::GUO_PRESSURE_OUTLET;

        if (y == 0 || y == NY-1 || z == 0 || z == NZ-1)
            return BC_flag::BOUNCE_BACK;

        return BC_flag::FLUID;
    }
};

/* ------------------------------------------------------------------ */
/*  Initial condition: plug flow + small noise                          */
/* ------------------------------------------------------------------ */
struct WindTunnelInit
{
    __device__ inline 
    void apply_forces(float* rho, float* u, float* force, int node) {
        force[get_vec_index(node, 0)] = 0.0f;
        force[get_vec_index(node, 1)] = 0.0f;
        force[get_vec_index(node, 2)] = 0.0f;
    }
    
    __device__ inline
    void operator()(float* rho, float* u, float* force, int node) const
    {
        rho[node] = 1.0f;

        u[get_vec_index(node, 0)] = 0.05f;   // prescribed bulk velocity
        u[get_vec_index(node, 1)] = 0.0f;
        u[get_vec_index(node, 2)] = 0.0f;

        force[get_vec_index(node, 0)] = 0.0f;
        force[get_vec_index(node, 1)] = 0.0f;
        force[get_vec_index(node, 2)] = 0.0f;
    }
};

/* ------------------------------------------------------------------ */
/*  Scenario trait                                                      */
/* ------------------------------------------------------------------ */
struct WindTunnelScenario : public ScenarioTrait<
        WindTunnelInit,
        WindTunnelBoundary,
        void,                       // no analytical solution
        CM<3, NoAdapter>,                     // or CM<3,NoAdapter> if you prefer
        NoAdapter
    >
{
    static constexpr float u_max     = 0.05f;
    static constexpr float Re_target = 1000.0f;   // not used here
    static constexpr float D         = NY;        // characteristic length
    static constexpr float viscosity = u_max * D / Re_target;
    static constexpr float tau       = viscosity_to_tau(viscosity);
    static constexpr float omega     = 1.0f / tau;

    // S matrix for MRT/CM (if you switch to CM)
    static constexpr float S[quadratures] = {
        1.0f, 1.0f, 1.0f, 1.0f, omega, omega, omega, omega, omega,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f
    };


    static const char* name() { return "WindTunnel3D"; }

    static InitType      init()      { return InitType(); }
    static BoundaryType  boundary()  { return BoundaryType(); }
    static ValidationType validation(){ return ValidationType(); }

    static void add_bodies() {
        std::string base_path = "";
        #ifdef _WIN32
            base_path = "../../../";
        #endif
        std::string sphere_config = base_path + "src/scenarios/windTest/sphere_config.ini";
        IBMBody sphere = conf::create_body_from_config(sphere_config);
        IBM_bodies.emplace_back(sphere);

    }
};

#endif // WIND_TUNNEL_SCENARIO_H
