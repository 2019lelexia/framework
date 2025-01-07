#pragma once
#include <vector>
#include <filesystem>
#include <gtsam/inference/Ordering.h>
#include <gtsam/geometry/Pose3.h>
#include <opencv4/opencv2/opencv.hpp>
#include "types.h"
#include "process_image.h"

class IMUInfo
{
public:
    IMUInfo(string _infoFilePath);
    ~IMUInfo();
    void readIMUInfo();

    string infoFilePath;
    SE3 T_cam_imu;
    double b_a_sigma = 0.00447213;
    double b_g_sigma = 0.0014142;
    double acc_sigma = 0.316227;
    double gyr_sigma = 0.1;
    double integration_sigma = 0.316227;
    gtsam::Vector3 gravity = (gtsam::Vector(3) << 0, 0, -9.8082).finished();
};

struct IMUDataUntreated
{
    double timestampsUntreated;
    Vector3d accUntreated;
    Vector3d gyrUntreated;
};

class IMUOneMeasurement
{
public:
    IMUOneMeasurement();
    IMUOneMeasurement(IMUDataUntreated _imuDataUntreated, double _integrationTime);
    ~IMUOneMeasurement();
    Vector3d getAcc();
    Vector3d getGyr();
    double getIntegrationTime();
private:
    Vector3d acc;
    Vector3d gyr;
    double integrationTime;
};

class IMUAlbum
{
public:
    IMUAlbum(string _imuFilePath);
    ~IMUAlbum();
    void readIMUData();
    void setTimestampsImage(shared_ptr<ImageFolder> _imageFolder);
    vector<IMUOneMeasurement> IMUData;
    string imuFilePath;
    shared_ptr<vector<double>> timestampsImage;
};