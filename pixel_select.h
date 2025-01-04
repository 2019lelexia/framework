#pragma once
#include <opencv4/opencv2/opencv.hpp>
#include <vector>
#include "frame.h"
using namespace std;

class PixelSelector
{
public:
    PixelSelector();
    ~PixelSelector();
    void setParameters(int _thresholdFast, int _widthGrid, int _heightGrid, int _edgeLeaveOutW, int _edgeLeaveOutH, int _numBlockW, int _numBlockH, int _numKeyTotal, int _numNormalTotal, int _nonMaxBlockEdge, int _thresholdSelectedGradient);
    

    int thresholdFast;
    int widthGrid;
    int heightGrid;
    int edgeLeaveOutW;
    int edgeLeaveOutH;
    int numBlockW;
    int numBlockH;
    int nonMaxBlockEdge;
    int thresholdSelectedGradient;
    vector<int> numKeyTotal;
    vector<int> numNormalTotal;
    void selectKeyPointFromImage(shared_ptr<Frame> &frame);
    void selectNormalPointFromImage(shared_ptr<Frame> &frame);
    void lookKeyPoint(shared_ptr<Frame> &frame);
    void lookNormalPoint(shared_ptr<Frame> &frame);
    
};