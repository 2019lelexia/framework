#include "imu_read.h"

IMUOneMeasurement::IMUOneMeasurement()
{
    acc.setZero();
    gyr.setZero();
    integrationTime = 0;
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