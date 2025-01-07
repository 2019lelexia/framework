#pragma once

#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include "global_params.h"
#include <vector>
#include <filesystem>

using namespace std;
class ImageInfo
{
public:
    ImageInfo();
    ~ImageInfo();
    void makePyramids();
    void readImage(string image_path);
    void setGlobalSize();
    void setGlobalCalibration(float fx, float fy, float cx, float cy);
    void calculateGrad();
    void calculateGradOrigin();



    cv::Mat image;
    vector<cv::Mat> dx, dy, grad_2;
    vector<cv::Mat> pyramids;

};

class ImageFolder
{
public:
    ImageFolder(string path_folder, string path_calibration, string path_timestamps);
    ~ImageFolder();
    void readImageFolder();
    void readTimestamps();
    shared_ptr<ImageInfo> getIndice(int index);


    string pathFolder;
    string pathCalibration;
    string pathTimestamps;
    vector<shared_ptr<ImageInfo>> album;
    vector<double> timestamps;
    vector<int> ids;
    int totalSize;
};