#pragma once
#ifndef IMMATURE_POINT_H
#define IMMATURE_POINT_H

#include <array>
#include "point.h"
#include "map_point.h"
#include "frame.h"

class MapPoint;
class Frame;

class Immature
{
public:
    Immature(shared_ptr<Frame> _source_frame, shared_ptr<MapPoint> _map_point);
    ~Immature();

    shared_ptr<MapPoint> map_point;
    float energyTH;
    array<float, PATTERNNUMFORINDEX> pixelValues;
    array<float, PATTERNNUMFORINDEX> pixelWeights;
    Matrix2f gradientHessian;

    float depthMin = 0;
    float depthMax = NAN;

};

#endif