#ifndef DEFINES_H
#define DEFINES_H

#define D3Q27
// #define D2Q9

// #define USE_TAYLOR_GREEN
// #define USE_POISEUILLE
// #define USE_LID_DRIVEN
// #define USE_TURBULENT_CHANNEL
// #define USE_FLOW_PAST_CYLINDER
 #define USE_TAYLOR_GREEN_3D
// #define USE_FLOW_PAST_SQUARE_CYLINDER
// #define USE_ZERO_FLOW_TEST
// #define USE_WIND_TUNNEL

// ---- Memory layout (pick at most one; default is SoA) ----
// #define LBM_LAYOUT_AOS
// #define LBM_LAYOUT_CSOA

// ---- Collision operator for TGV benchmarks (pick at most one; default is CM) ----
// #define LBM_COLLISION_BGK
// #define LBM_COLLISION_MRT
// #define LBM_COLLISION_MRT_BASELINE   // slow MRT + f_eq (baseline; use with AoS)
#if (defined(LBM_COLLISION_BGK) + defined(LBM_COLLISION_MRT) + defined(LBM_COLLISION_MRT_BASELINE)) > 1
#error "Pick at most one collision operator define"
#endif

// simulation parameters
#ifdef USE_TAYLOR_GREEN
    #define SCALE 2
    #define NX (128 * SCALE)
    #define NY (128 * SCALE)
#elif defined(USE_POISEUILLE)
    #define SCALE 1
    #define NX (150 * SCALE)
    #define NY (100 * SCALE)
#elif defined(USE_LID_DRIVEN)
    #define SCALE 1
    #define NX (129 * SCALE)
    #define NY (129 * SCALE)
#elif defined(USE_FLOW_PAST_CYLINDER)
    #define SCALE 1
    #define NX (256 * SCALE)
    #define NY (128 * SCALE)
#elif defined(USE_TAYLOR_GREEN_3D)
    #define SCALE 2
    #define NX (64 * SCALE)
    #define NY (64 * SCALE)
    #define NZ (64 * SCALE)
#elif defined(USE_TURBULENT_CHANNEL)
    #define SCALE 1
    #define NX (256 * SCALE)
    #define NY (128 * SCALE)
    #define NZ (256 * SCALE)
#elif defined(USE_FLOW_PAST_SQUARE_CYLINDER)
    #define SCALE 1
    #define NX (512/2 * SCALE)
    #define NY (128/2 * SCALE)
    #define NZ (96/2 * SCALE)
#elif defined(USE_ZERO_FLOW_TEST)
    #define SCALE 1
    #define NX (512/4 * SCALE)
    #define NY (128/4 * SCALE)
    #define NZ (96/4 * SCALE)
#elif defined(USE_WIND_TUNNEL)
    #define SCALE 1
    #define NX (384 * SCALE)
    #define NY (196 * SCALE)
    #define NZ (196  * SCALE)
#endif


#ifndef NZ
    #define NZ 1
#endif


// CUDA specific stuff
#define BLOCK_SIZE 16

// Debug/diagnostic configuration
// #define LBM_DEBUG
#ifndef DEBUG_NODE
  #define DEBUG_NODE 5
#endif
#ifndef VALUE_THRESHOLD
  #define VALUE_THRESHOLD 5.0f
#endif

#endif // !DEFINES_H
