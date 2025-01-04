#pragma once
#include <eigen3/Eigen/Core>
#include "types.h"

class AffineLight
{
public:
    AffineLight();
    AffineLight(float _a, float _b);
    ~AffineLight();
    // a, b belong to every frame, but we often use the deviation like I2 - (A * I1 - B), here we return A and B
    static Vector2d deviationAffineLight(float _exposure1, float _exposure2, AffineLight aff1, AffineLight aff2)
    {
        if(_exposure1 == _exposure2 == 0)
        {
            _exposure1 = _exposure2 = 1;
        }
        float A = exp(aff2.a - aff1.a) * _exposure2 / _exposure1;
        float B = aff2.b - A * aff1.b;
        return Vector2d(A, B);
    }

    float a;
    float b;
};