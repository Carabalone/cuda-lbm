#include "core/collision/CM/CM.cuh"

template struct CM<2, NoAdapter>;
template struct CM<2, OptimalAdapter>;

template struct CM<3, NoAdapter>;
template struct CM<3, OptimalAdapter>;
template struct CM<3, EmpiricalAdapter>;