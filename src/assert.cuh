#ifndef LBM_ASSERT_H
#define LBM_ASSERT_H

#include <stdio.h>
#include <cassert>

#define LBM_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            printf("[ASSERT FAILED] %s:%d - %s\nCondition: %s\n", \
                    __FILE__, __LINE__, message, #condition); \
            assert(condition); \
        } \
    } while(0)

#define LBM_DEVICE_ASSERT(condition, message) \
        do { \
            if (!(condition)) { \
                printf("[DEVICE ASSERT FAILED] Block [%d,%d,%d], Thread [%d,%d,%d] - %s:%d - %s\n", \
                       blockIdx.x, blockIdx.y, blockIdx.z, \
                       threadIdx.x, threadIdx.y, threadIdx.z, \
                       __FILE__, __LINE__, message); \
                assert(condition); \
            } \
        } while(0)

#endif // LBM_ASSERT_H