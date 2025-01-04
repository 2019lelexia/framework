#pragma once
#ifndef FRAME_H
#define FRAME_H

#include "process_image.h"
#include "point.h"
#include "light_affine.h"
#include "nanoflann.hpp"
#include <pcl/point_cloud.h>
#include <pcl/cloud_iterator.h>
#include <pcl/visualization/pcl_visualizer.h>

using namespace nanoflann;

class Point;

class Frame
{
public:
    Frame();
    ~Frame();
    void setFrame(shared_ptr<ImageInfo> _image);
    void transformToConcern();
    Vector3f interpolationPixelDxDy(int level, float u, float v);
    float interpolationPixel(int level, float u, float v);
    void makePointCloudFlann();
    void visualizePointCloudLevel(int level);
    void visualizePointCloudAllLevel();


    shared_ptr<ImageInfo> image;
    vector<vector<cv::KeyPoint>> keyPoints;
    vector<cv::Mat> keyPointsMask;
    vector<vector<cv::Point>> normalPoints;
    vector<cv::Mat> normalPointsMask;

    vector<vector<shared_ptr<Point>>> concernKeyPoints;
    vector<vector<shared_ptr<Point>>> concernNormalPoints;
    vector<int> numPoints;

    AffineLight affine;
    float exposure;
};

struct myPointCloudFlann
{
    int num;
    vector<shared_ptr<Point>> points;
    myPointCloudFlann(int _num, const vector<shared_ptr<Point>> &_points) : num(_num), points(_points){}
    inline size_t kdtree_get_point_count() const
    {
        return points.size();
    }
    inline float kdtree_get_pt(const size_t index, const size_t dim) const;
    inline float kdtree_distance(const float *p1, const size_t index_p2, size_t ) const;
    template<class BBOX>
    bool kdtree_get_bbox(BBOX& ) const
    {
        return false;
    }
};

#endif