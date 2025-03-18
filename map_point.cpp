#include "map_point.h"


MapPoint::MapPoint(shared_ptr<Frame> _frame, float _u, float _v)
{
    hostFrameOfMapPoint = _frame;
    positionX = _u;
    positionY = _v;
    state = 0;
    isKeyPoint = false;
}

MapPoint::~MapPoint()
{}

void MapPoint::createFromImmaturePoint()
{
    // w/o the definition of Point Class of DSO, so there is no need to create others, just as PointHessian ctor
    if(immaturePoint == nullptr)
    {
        cerr << "no immaturePoint before create map point from it" << endl;
        exit(0);
    }
    depth = (immaturePoint->depthMax + immaturePoint->depthMin) / 2;
    depthWithScale = depth;
    pixelMapValues = immaturePoint->pixelValues;
    pixelMapWeights = immaturePoint->pixelWeights;
    energyTH = immaturePoint->energyTH;
    state = 1;
}

void MapPoint::deleteThisPoint()
{
    immaturePoint->map_point = nullptr;
    immaturePoint = nullptr;
}

void MapPoint::setDepth(float _depth)
{
    depth = _depth;
    depthWithScale = _depth;
    inverseDepth = depth;
}

void MapPoint::setDepthScaled(float _depthScaled)
{
    depth = _depthScaled;
    depthWithScale = _depthScaled;
    inverseDepth = depth;
}

void MapPoint::setDepthZero(float _depthZero)
{
    depthZero = _depthZero;
    depthZeroWithScale = _depthZero;
    nullspaceScale = -(depth * 1.001 - depth / 1.001) * 500;
}