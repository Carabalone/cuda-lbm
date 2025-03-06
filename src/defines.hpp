#ifndef DEFINES_H
#define DEFINES_H

// #define USE_TAYLOR_GREEN
// #define USE_POISEUILLE
#define USE_LID_DRIVEN
// #define USE_TURBULENT_CHANNEL

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
#elif defined(USE_TURBULENT_CHANNEL)
    #define SCALE 1
    #define NX (512 * SCALE)
    #define NY (256 * SCALE)
#endif

// #define D3Q27
#define D2Q9

// CUDA specific stuff
#define BLOCK_SIZE 16

#endif // !DEFINES_H
