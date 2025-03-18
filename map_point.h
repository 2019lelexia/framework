#pragma once
#ifndef MAP_POINT_H
#define MAP_POINT_H

#include <opencv4/opencv2/opencv.hpp>
#include "frame.h"
#include "immature_point.h"

class Frame;
class Immature;

class MapPoint
{
public:
    enum PointStatus
    {
        ACTIVE = 0,     // point is still active in optimization
        OUTLIER,        // considered as outlier in optimization
        OUT,            // out side the boundary
        MARGINALIZED    // marginalized, usually also out of boundary, but also can be set because the host frame is marged
    };
    MapPoint(shared_ptr<Frame> _frame, float _u, float _v);
    ~MapPoint();

    void createFromImmaturePoint();
    void deleteThisPoint();
    void setDepth(float _depth);
    void setDepthScaled(float _depthScaled);
    void setDepthZero(float _depthZero);
    
    weak_ptr<Frame> hostFrameOfMapPoint;
    shared_ptr<Immature> immaturePoint;
    float positionX;
    float positionY;
    bool isKeyPoint;
    float inverseDepth = -1; // invD of feature
    int state; // immature = 0, good = 1, outlier = 2

    // things below belong to PointHessian
    float depth = -1;
    float depthWithScale = -1;
    float depthZero = -1;
    float depthZeroWithScale = -1;
    float nullspaceScale = -1;
    float energyTH = 0;
    array<float, PATTERNNUMFORINDEX> pixelMapValues;
    array<float, PATTERNNUMFORINDEX> pixelMapWeights;

    bool ownDepthPrior = false;
    PointStatus stateOptimize = PointStatus::ACTIVE;

    float prior = 0;
    float delta = 0;
};


#endif