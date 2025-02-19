#ifndef DEFINES_H
#define DEFINES_H

// simulation parameters
// #define NX 400
// #define NY 200
#define SCALE 2
#define NX (128 * SCALE)
#define NY (128 * SCALE)

// #define D3Q27
#define D2Q9

// CUDA specific stuff
#define BLOCK_SIZE 16

#endif // !DEFINES_H
