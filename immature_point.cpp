#include "immature_point.h"

Immature::Immature(shared_ptr<Frame> _source_frame, shared_ptr<MapPoint> _map_point): map_point(_map_point)
{
    shared_ptr<Frame> ptr_frame = _source_frame;
    float positionX = map_point->positionX;
    float positionY = map_point->positionY;
    for(int index = 0; index < patternNum; index++)
    {
        int dx = pattern.at(index).first;
        int dy = pattern.at(index).second;
        Vector3f pointValue = ptr_frame->interpolationPixelAndCalculateDxDy(0, positionX, positionY);
        pixelValues.at(index) = pointValue[0];
        if(!isfinite(pixelValues.at(index)))
        {
            energyTH = NAN;
            return;
        }
        gradientHessian += pointValue.tail<2>() * pointValue.tail<2>().transpose();
        pixelWeights.at(index) = sqrtf(2500 / (2500 + pointValue.tail<2>().squaredNorm()));
    }
    energyTH = patternNum * outlierEnergyThreshold;
}

Immature::~Immature()
{}