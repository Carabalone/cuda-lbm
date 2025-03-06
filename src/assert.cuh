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

#endif // LBM_ASSERT_H