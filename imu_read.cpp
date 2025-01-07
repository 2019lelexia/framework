#include "imu_read.h"

IMUInfo::IMUInfo(string _infoFilePath)
{
    infoFilePath = _infoFilePath;
}

IMUInfo::~IMUInfo()
{}

void IMUInfo::readIMUInfo()
{
    cv::FileStorage fs(infoFilePath, cv::FileStorage::READ);
    if(!fs.isOpened())
    {
        cerr << "Failed to open imu info file." << endl;
        exit(1);
    }
    cv::Mat tmpT;
    fs["T_cam_imu"] >> tmpT;
    cv::Mat tmpR = tmpT(cv::Rect(0, 0, 3, 3));
    cv::Mat tmpt = tmpT(cv::Rect(3, 0, 1, 3));
    T_cam_imu = SE3(Eigen::Matrix3d(tmpR), Eigen::Matrix3d(tmpt));
    fs["imu/accelerometer_noise_density"] >> acc_sigma;
    fs["imu/gyroscope_noise_density"] >> gyr_sigma;
    fs["imu/accelerometer_random_walk"] >> b_a_sigma;
    fs["imu/gyroscope_random_walk"] >> b_g_sigma;
    fs["imu/integration_sigma"] >> integration_sigma;
    fs.release();
}

IMUOneMeasurement::IMUOneMeasurement()
{
    acc.setZero();
    gyr.setZero();
    integrationTime = 0;
}

IMUOneMeasurement::IMUOneMeasurement(IMUDataUntreated _imuDataUntreated, double _integrationTime)
{
    acc = _imuDataUntreated.accUntreated;
    gyr = _imuDataUntreated.gyrUntreated;
    integrationTime = _integrationTime;
}

IMUOneMeasurement::~IMUOneMeasurement()
{}

Vector3d IMUOneMeasurement::getAcc()
{
    return acc;
}

Vector3d IMUOneMeasurement::getGyr()
{
    return gyr;
}

double IMUOneMeasurement::getIntegrationTime()
{
    return integrationTime;
}

IMUAlbum::IMUAlbum(string _imuFilePath)
{
    imuFilePath = _imuFilePath;
    IMUData.clear();
}

IMUAlbum::~IMUAlbum()
{}

void IMUAlbum::setTimestampsImage(shared_ptr<ImageFolder> _imageFolder)
{
    timestampsImage = make_shared<vector<double>>(_imageFolder->timestamps);
}

void IMUAlbum::readIMUData()
{
    vector<IMUDataUntreated> totalIMUData;
    ifstream imuStream(imuFilePath);
    if(!imuStream.is_open())
    {
        cerr << "Failed to open imu file." << endl;
        return;
    }
    string line;
    double tmpTime, tmpWx, tmpWy, tmpWz, tmpAx, tmpAy, tmpAz;
    while(getline(imuStream, line))
    {
        if(line[0] == '#')
        {
            continue;
        }
        stringstream lineStream(line);
        IMUDataUntreated imuUntreated;
        lineStream >> imuUntreated.timestampsUntreated >> imuUntreated.gyrUntreated.x() >> imuUntreated.gyrUntreated.y() >> imuUntreated.gyrUntreated.z() >> imuUntreated.accUntreated.x() >> imuUntreated.accUntreated.y() >> imuUntreated.accUntreated.z();
        totalIMUData.push_back(imuUntreated);
    }
    imuStream.close();
    
}