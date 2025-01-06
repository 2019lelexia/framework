#include "process_image.h"

ImageInfo::ImageInfo()
{}

ImageInfo::~ImageInfo()
{}

void ImageInfo::readImage(string image_path)
{
    cv::Mat tmp = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    tmp.convertTo(image, CV_32F);
    if(wG.size() == 0 || hG.size() == 0)
    {
        if(image.cols == 0 || image.rows == 0)
        {
            cout << "no image" << endl;
            exit(1);
        }
    }
    else
    {
        if(image.cols != wG.at(0) || image.rows != hG.at(0))
        {
            cout << "inconsistent size" << endl;
            exit(1);
        }
    }
}

void ImageInfo::makePyramids()
{
    pyramids.push_back(image);
    for(int i = 0; i < LEVEL - 1; i++)
    {
        cv::Mat source = pyramids.at(i);
        cv::Mat target;
        cv::resize(source, target, cv::Size(source.cols / 2, source.rows / 2), 0, 0, cv::INTER_LINEAR);
        pyramids.push_back(target);
    }
}

void ImageInfo::setGlobalSize()
{
    for(int i = 0; i < LEVEL; i++)
    {
        int wi = pyramids.at(i).cols;
        int hi = pyramids.at(i).rows;
        wG.push_back(wi);
        hG.push_back(hi);
    }
}

void ImageInfo::setGlobalCalibration(float fx, float fy, float cx, float cy)
{
    float tmpFx = fx;
    float tmpFy = fy;
    float tmpCx = cx;
    float tmpCy = cy;
    for(int i = 0; i < LEVEL; i++)
    {
        fxG.push_back(tmpFx);
        fyG.push_back(tmpFy);
        cxG.push_back(tmpCx);
        cyG.push_back(tmpCy);
        tmpFx /= 2;
        tmpFy /= 2;
        tmpCx /= 2;
        tmpCy /= 2;
    }
    for(int i = 0; i < LEVEL; i++)
    {
        Matrix3d tmpK;
        tmpK << fxG[i], 0.0, cxG[i], 0.0, fyG[i], cyG[i], 0.0, 0.0, 1.0;
        KG.push_back(tmpK);
        KinvG.emplace_back(KG[i].inverse());
    }
}

void ImageInfo::calculateGrad()
{
    for(int i = 0; i < LEVEL; i++)
    {
        cv::Mat tmpdx, tmpdy, tmpgrad_2;
        cv::Mat xKernel = (cv::Mat_<float>(3, 3) << 0, 0, 0, -0.5, 0, 0.5, 0, 0, 0);
        cv::Mat yKernel = (cv::Mat_<float>(3, 3) << 0, -0.5, 0, 0, 0, 0, 0, 0.5, 0);

        
        // cv::Sobel(pyramids.at(i), tmpdx, CV_32F, 1, 0, 3);
        // cv::Sobel(pyramids.at(i), tmpdy, CV_32F, 0, 1, 3);
        cv::filter2D(pyramids.at(i), tmpdx, CV_32F, xKernel);
        cv::filter2D(pyramids.at(i), tmpdy, CV_32F, yKernel);
        cv::magnitude(tmpdx, tmpdy, tmpgrad_2);
        dx.push_back(tmpdx);
        dy.push_back(tmpdy);
        grad_2.push_back(tmpgrad_2);
    }
}

void ImageInfo::calculateGradOrigin()
{
    for(int i = 0; i < LEVEL; i++)
    {
        cv::Mat tmp = pyramids.at(i);
        cv::Mat tmpdx = cv::Mat::zeros(tmp.size(), CV_32F);
        cv::Mat tmpdy = cv::Mat::zeros(tmp.size(), CV_32F);
        cv::Mat tmpgrad_2 = cv::Mat::zeros(tmp.size(), CV_32F);
        int w = wG[i];
        int h = hG[i];
        for(int index = w; index < w * (h - 1); index++)
        {
            float deltax = 0.5f * (tmp.at<float>(index + 1) - tmp.at<float>(index - 1));
            float deltay = 0.5f * (tmp.at<float>(index + w) - tmp.at<float>(index - w));
            if(isnan(deltax) || fabs(deltax) > 255.0)
            {
                deltax = 0;
            }
            if(isnan(deltay) || fabs(deltay) > 255.0)
            {
                deltay = 0;
            }
            tmpdx.at<float>(index) = deltax;
            tmpdy.at<float>(index) = deltay;
            tmpgrad_2.at<float>(index) = deltax * deltax + deltay * deltay;
        }
        dx.push_back(tmpdx);
        dy.push_back(tmpdy);
        grad_2.push_back(tmpgrad_2);
    }
}


ImageFolder::ImageFolder(string path_folder, string path_calibration): pathFolder(path_folder), pathCalibration(path_calibration), totalSize(0)
{}

ImageFolder::~ImageFolder()
{}

void ImageFolder::readImageFolder()
{
    vector<std::filesystem::path> pathes;
    for(const auto& entry: std::filesystem::directory_iterator(pathFolder))
    {
        const auto& path = entry.path();
        if(path.extension() == ".jpg" || path.extension() == ".png")
        {
            pathes.push_back(path);
        }
    }
    sort(pathes.begin(), pathes.end());

    for(const auto& path: pathes)
    {
        shared_ptr<ImageInfo> ptr_imageInfo = make_shared<ImageInfo>();
        ptr_imageInfo->readImage(path.string());
        ptr_imageInfo->makePyramids();
        if(totalSize == 0)
        {
            cv::FileStorage fs(pathCalibration, cv::FileStorage::READ);
            if (!fs.isOpened())
            {
                cerr << "Failed to open the file." << endl;
                return;
            }
            ptr_imageInfo->setGlobalCalibration(fs["fx"], fs["fy"], fs["cx"], fs["cy"]);
            ptr_imageInfo->setGlobalSize();
        }
        ptr_imageInfo->calculateGradOrigin();
        album.push_back(ptr_imageInfo);
        ids.push_back(totalSize);
        totalSize++;
    }
    assert(album.size() == totalSize);
}

shared_ptr<ImageInfo> ImageFolder::getIndice(int index)
{
    if(index >= totalSize)
    {
        cout << "[ImageFolder] out of totalSize" << endl;
        exit(1);
    }
    else
    {
        return album.at(index);
    }
}