#include "trajectory.h"

using namespace std;

Trajectory::Trajectory()
{
    idFrames.push_back(0);
    posesFrame.emplace_back(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
}

Trajectory::~Trajectory()
{}

void Trajectory::addPoseAndAffineOfTrajectory(int _idFrame, SE3 _poseOfThisFrame, AffineLight _affineThisFrame)
{
    idFrames.push_back(_idFrame);
    posesFrame.push_back(_poseOfThisFrame);
    affinesFrame.push_back(_affineThisFrame);
}

void Trajectory::visualizeTrajectory()
{
    vector<Eigen::Isometry3d> poses;
    for(int i = 0; i < posesFrame.size(); i++)
    {
        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        pose.translate(posesFrame.at(i).translation() * 100);
        cout << posesFrame.at(i).translation().transpose() << endl;
        poses.push_back(pose);
    }
    pcl::visualization::PCLVisualizer viewer("Trajectory Viewer");
    viewer.setBackgroundColor(0.0, 0.0, 0.0);
    viewer.addCoordinateSystem(1.0);
    viewer.setCameraPosition(0, 0, 10, 0, 0, 0);
    for (size_t i = 1; i < poses.size(); ++i)
    {
        Vector3d t1 = poses[i - 1].translation();
        Vector3d t2 = poses[i].translation();
        string line_id = "line_" + to_string(i);
        viewer.addLine<pcl::PointXYZ>(pcl::PointXYZ(t1.x(), t1.y(), t1.z()),
                                      pcl::PointXYZ(t2.x(), t2.y(), t2.z()), 255, 0, 0, line_id);
    }
    while (!viewer.wasStopped())
    {
        viewer.spinOnce(100);
    }
}