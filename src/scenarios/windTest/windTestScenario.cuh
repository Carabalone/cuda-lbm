#ifndef WIND_TUNNEL_SCENARIO_H
#define WIND_TUNNEL_SCENARIO_H

#include "scenarios/scenario.cuh"
#include "functors/initialConditions/defaultInit.cuh"
#include "IBM/config/body_config.hpp"
#include "util/utility.cuh"

struct WindTunnelBoundary
{
    __host__ __device__
    int operator()(int x, int y, int z) const
    {
        const int edges = (x==0) + (x==NX-1) +
                          (y==0) + (y==NY-1) +
                          (z==0) + (z==NZ-1);

        // if (edges >= 2) return BC_flag::EXTRAPOLATED_CORNER_EDGE;

        // if (x == 0)       return BC_flag::GUO_VELOCITY_INLET;
        if (x == 0)       return BC_flag::REGULARIZED_INLET_LEFT;
        // if (x == 0)       return BC_flag::BOUNCE_BACK;

        // if (x == NX-1)    return BC_flag::PRESSURE_OUTLET;
        // if (x == NX-1)    return BC_flag::GUO_PRESSURE_OUTLET;
        // if (x == NX-1)    return BC_flag::CONVECTIVE_OUTLET_RIGHT;
        if (x == NX-1)    return BC_flag::ZG_OUTFLOW;
        // if (x == NX-1)    return 150;
        // if (x == NX-1)    return BC_flag::BOUNCE_BACK;

        if (y == 0 || y == NY-1 || z == 0 || z == NZ-1)
            return BC_flag::BOUNCE_BACK;
            // return BC_flag::FREE_SLIP_BOUNCE_BACK;

        return BC_flag::FLUID;
    }
};

struct WindTunnelInit
{
    float u_max;

    __host__ __device__
    WindTunnelInit(float u_max) : u_max(u_max) { }

    __device__ inline 
    void apply_forces(float* rho, float* u, float* force, int node) {
        force[get_vec_index(node, 0)] = 0.0f;
        force[get_vec_index(node, 1)] = 0.0f;
        force[get_vec_index(node, 2)] = 0.0f;
    }
    
    __device__ inline
    void operator()(float* rho, float* u, float* force, int node)
    {
        int x, y, z;
        get_coords_from_node(node, x, y, z);
        
        float y_center = (NY - 1) * 0.5f;
        float z_center = (NZ - 1) * 0.5f;
        float r_y = (y - y_center) / y_center;
        float r_z = (z - z_center) / z_center;
        float r2 = r_y*r_y + r_z*r_z;
        
        u[get_vec_index(node, 0)] = u_max * (1.0f - r2) * 1.0f;  // 80% of final
        u[get_vec_index(node, 1)] = 0.0f;
        u[get_vec_index(node, 2)] = 0.0f;
        
        rho[node] = 1.0f;
        apply_forces(rho, u, force, node);
    }
};

/* ------------------------------------------------------------------ */
/*  Scenario trait                                                      */
/* ------------------------------------------------------------------ */
struct WindTunnelScenario : public ScenarioTrait<
        WindTunnelInit,
        WindTunnelBoundary,
        void,
        CM<3, NoAdapter>,
        NoAdapter
    >
{
    static constexpr float u_max      = 0.05f;
    static constexpr float Re_target  = 5000.0f;
    static constexpr float D          = NY;
    static constexpr float viscosity  = u_max * (D * 0.5f) / Re_target;
    static constexpr float tau        = viscosity_to_tau(viscosity);
    static constexpr float omega      = 1.0f / tau;

    static constexpr int   ramp_up_time     = 0;
    static constexpr int   sponge_layers    = 20;
    static constexpr float sponge_sigma_max = 0.1f;
    static constexpr float sponge_rho = 1.0f;
    static constexpr float sponge_u   = 0.0f;

    static constexpr float S[quadratures] = {
        1.0f, 1.0f, 1.0f, 1.0f, omega, omega, omega, omega, omega,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f
    };


    static const char* name() { return "WindTunnel3D"; }

    static InitType      init()      { return InitType(u_max); }
    static BoundaryType  boundary()  { return BoundaryType(); }
    static ValidationType validation(){ return ValidationType(); }

    static void add_bodies() {
        std::string base_path = "";
        #ifdef _WIN32
            base_path = "../../../";
        #endif
        std::string sphere_config = base_path + "src/scenarios/windTest/sphere_config.ini";
        // IBMBody sphere = conf::create_body_from_config(sphere_config);
        // IBM_bodies.emplace_back(sphere);

    }
};

#endif // WIND_TUNNEL_SCENARIO_H
