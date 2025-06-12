#ifndef DEFINES_H
#define DEFINES_H

#define D3Q27
// #define D2Q9

// #define USE_TAYLOR_GREEN
// #define USE_POISEUILLE
// #define USE_LID_DRIVEN
// #define USE_TURBULENT_CHANNEL
// #define USE_FLOW_PAST_CYLINDER
// #define USE_TAYLOR_GREEN_3D
// #define USE_FLOW_PAST_SQUARE_CYLINDER
#define USE_ZERO_FLOW_TEST

// #define SOA

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
    #define NX (512 * SCALE)
    #define NY (128 * SCALE)
    #define NZ (96 * SCALE)
#elif defined(USE_ZERO_FLOW_TEST)
    #define SCALE 1
    #define NX (512 * SCALE)
    #define NY (128 * SCALE)
    #define NZ (96 * SCALE)
#endif

#ifndef NZ
    #define NZ 1
#endif


// CUDA specific stuff
#define BLOCK_SIZE 16

#endif // !DEFINES_H
