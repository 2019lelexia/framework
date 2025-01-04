#include "light_affine.h"

AffineLight::AffineLight()
{
    a = b = 0;
}

AffineLight::AffineLight(float _a, float _b) : a(_a), b(_b)
{}

AffineLight::~AffineLight()
{}