#pragma once
#ifndef MAP_H
#define MAP_H

#include <opencv4/opencv2/opencv.hpp>
#include "frame.h"
#include "nanoflann.hpp"

using namespace std;
class Map
{
public:
    Map();
    ~Map();
    void addKeyFrameToMap(shared_ptr<Frame> _frame);

    struct CompareFrameid
    {
        bool operator()(const std::shared_ptr<Frame>& f1, const std::shared_ptr<Frame>& f2) const
        {
            return f1->id < f2->id;
        }
    };
    std::set<std::shared_ptr<Frame>, CompareFrameid> frames;
};

#endif