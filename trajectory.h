#pragma once
#include "tracker.h"
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <iostream>

class Trajectory
{
public:
    Trajectory();
    ~Trajectory();
    void addPoseAndAffineOfTrajectory(int _idFrame, SE3 _poseOfThisFrame, AffineLight _affineThisFrame);
    void visualizeTrajectory();

    vector<shared_ptr<Frame>> trackingFrames;
    vector<int> idFrames;
    vector<SE3> posesFrame;
    vector<AffineLight> affinesFrame;
    vector<Sim3> posesFusion;
};