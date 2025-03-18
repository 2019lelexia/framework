#pragma once
#ifndef FRAME_H
#define FRAME_H

#include "process_image.h"
#include "point.h"
#include "light_affine.h"
#include "map_point.h"
#include "nanoflann.hpp"
#include <pcl/point_cloud.h>
#include <pcl/cloud_iterator.h>
#include <pcl/visualization/pcl_visualizer.h>

using namespace nanoflann;

class Point;
class MapPoint;

class Frame
{
public:
    Frame();
    ~Frame();
    static int global_id;
    int id;
    void setFrame(shared_ptr<ImageInfo> _image);
    void transformToConcern();
    Vector3f interpolationPixelDxDy(int level, float u, float v);
    Vector3f interpolationPixelAndCalculateDxDy(int level, float u, float v);
    float interpolationPixel(int level, float u, float v);
    void makePointCloudFlann();
    void visualizePointCloudLevel(int level, const SE3& transform);
    void visualizePointCloudAllLevel();
    void setState(const Vector10d &_state);
    void setStateScaled(const Vector10d &_state_scaled);
    void setStateZero(const Vector10d &_state_zero);
    void setPoseAndState(const SE3 &_Tw_to_c, const Vector10d &_state);
    void setPoseAndStateScaledInitially(const SE3 &_Tw_to_c, const AffineLight &_affine);


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
    vector<shared_ptr<MapPoint>> mapPoints;

    // things below belong to FrameHessian
    SE3 world_to_cam_linear;

    Matrix6d nullspacePose = Matrix6d::Zero();
    Matrix42d nullspaceAffine = Matrix42d::Zero();
    Vector6d nullspaceScale = Vector6d::Zero();

    SE3 Tw_to_c;
    Sim3 Tw_to_c_optimized;

    SE3 precalculate_Tw_to_c;
    SE3 precalculate_Tc_to_w;
    Vector10d state;
    Vector10d step = Vector10d::Zero();
    Vector10d stepBackup = Vector10d::Zero();
    Vector10d stateBackup = Vector10d::Zero();
    Vector10d stateZero = Vector10d::Zero();
    Vector10d stateScaled = Vector10d::Zero();
    
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