#include "point.h"

Point::Point(int _positionX, int _positionY, int _level, shared_ptr<Frame> _frame)
{
    positionX = _positionX;
    positionY = _positionY;
    level = _level;
    keyOrNormal = false;
    setInitialParameters();
    hostFrame = _frame;
}

Point::Point(cv::Point _point, int _level, shared_ptr<Frame> _frame)
{
    positionX = _point.x;
    positionY = _point.y;
    level = _level;
    keyOrNormal = false;
    setInitialParameters();
    hostFrame = _frame;
}

Point::Point(cv::KeyPoint _keyPoint, int _level, shared_ptr<Frame> _frame)
{
    positionX = _keyPoint.pt.x;
    positionY = _keyPoint.pt.y;
    level = _level;
    keyOrNormal = true;
    setInitialParameters();
    hostFrame = _frame;
}

Point::~Point()
{}

void Point::setInitialParameters()
{
    depth = 1;
    depthNew = depth;
    depthConvergence = 1;
    childNum = 0;
    variance = 0;
    varianceNew = variance;
    capability = 999;
    status = 0;
    isGood = true;
    energy.setZero();

    bAlpha = 0;
    JAlphaT_mul_JAlphaSingle = 1;
    JAlphaT_mul_JBeta = Vector8f::Zero();
    bAlphaNew = 0;
    JAlphaT_mul_JAlphaSingleNew = 1;
    JAlphaT_mul_JBetaNew = Vector8f::Zero();
    
    outlierThreshold = 8 * 12 * 12;
}

float Point::getPositionX()
{
    return positionX;
}

float Point::getPositionY()
{
    return positionY;
}