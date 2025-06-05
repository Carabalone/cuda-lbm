#ifndef MATH_CONSTANTS_H
#define MATH_CONSTANTS_H

constexpr float PI = 3.14159265358979323846f;

float inline deg2rad(float degs) {
  return degs * (PI / 180.0f);
}

float inline rad2deg(float rads)  {
  return rads * (180 / PI);
}

#endif // ! MATH_CONSTANTS_H
