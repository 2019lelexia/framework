#pragma once
#ifndef POINT_H
#define POINT_H

#include <opencv4/opencv2/opencv.hpp>
#include "frame.h"
#include "nanoflann.hpp"

class Frame;
// struct myPointCloudFlann;

class Point
{
public:
    Point(int _positionX, int _positionY, int _level, shared_ptr<Frame> _frame);
    Point(cv::Point _point, int _level, shared_ptr<Frame> _frame);
    Point(cv::KeyPoint _keyPoint, int _level, shared_ptr<Frame> _frame);
    ~Point();
    void setInitialParameters();
    float getPositionX();
    float getPositionY();
    // void setHost(weak_ptr<Frame> hostFrame);
    
    weak_ptr<Frame> hostFrame;
    int positionX;
    int positionY;
    int level;
    bool keyOrNormal;
    
    float depth;
    float depthNew;
    float variance;
    float varianceNew;

    float depthConvergence;
    int childNum;
    
    float capability;
    char status; // 0 - never ever used  1 - under optimization  2 - optimized  3 - dropped

    array<int, 5> indexNeighbour;
    array<float, 5> distanceNeighbour;

    int indexParent;
    float distanceParent;

    float bAlpha;
    float JAlphaT_mul_JAlphaSingle;
    Vector8f JAlphaT_mul_JBeta;
    float bAlphaNew;
    float JAlphaT_mul_JAlphaSingleNew;
    Vector8f JAlphaT_mul_JBetaNew;
    

    Vector2f energy;
    Vector2f energyNew;
    bool isGood;
    bool isGoodNew;

    float maxstep;
    float outlierThreshold;
};

#endif