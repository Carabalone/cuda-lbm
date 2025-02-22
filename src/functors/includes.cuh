#ifndef FUNCTOR_INCLUDES_H
#define FUNCTOR_INCLUDES_H

#include "./boundaryConditions/cornerBoundaries.cuh"
#include "./boundaryConditions/bbDomainBoundary.cuh"
#include "./boundaryConditions/cylinderBoundary.cuh"
#include "./boundaryConditions/zeroGradientOutflow.cuh"
#include "./boundaryConditions/zouHeInflow.cuh"
#include "./initialConditions/defaultInit.cuh"
#include "./initialConditions/taylorGreenInit.cuh"

#endif // ! FUNCTOR_INCLUDES_H