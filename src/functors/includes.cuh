#ifndef FUNCTOR_INCLUDES_H
#define FUNCTOR_INCLUDES_H

#include "./boundaryConditions/cornerBoundaries.cuh"
#include "./boundaryConditions/bbDomainBoundary.cuh"
#include "./boundaryConditions/cylinderBoundary.cuh"
#include "./boundaryConditions/zeroGradientOutflow.cuh"
#include "./boundaryConditions/pressureOutlet.cuh"
#include "./boundaryConditions/zouHeInflow.cuh"
#include "./boundaryConditions/regularizedInlet.cuh"
#include "./boundaryConditions/regularizedBounceBack.cuh"
#include "./boundaryConditions/regularizedBoundary3D.cuh"
#include "./boundaryConditions/edgeCornerBounceBack.cuh"
#include "./boundaryConditions/guoInlet3D.cuh"
#include "./boundaryConditions/guoOutlet3D.cuh"
#include "./boundaryConditions/regularizedOutlet.cuh"
#include "./initialConditions/defaultInit.cuh"

#endif // ! FUNCTOR_INCLUDES_H
