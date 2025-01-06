#pragma once
#include <vector>
#include "types.h"

class IMUOneMeasurement
{
public:
    IMUOneMeasurement();
    ~IMUOneMeasurement();
    Vector3d getAcc();
    Vector3d getGyr();
    double getIntegrationTime();
private:
    Vector3d acc;
    Vector3d gyr;
    double integrationTime;
};