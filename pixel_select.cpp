#include "pixel_select.h"

PixelSelector::PixelSelector()
{}

PixelSelector::~PixelSelector()
{}

void PixelSelector::setParameters(int _thresholdFast, int _widthGrid, int _heightGrid, int _edgeLeaveOutW, int _edgeLeaveOutH, int _numBlockW, int _numBlockH, int _numKeyTotal, int _numNormalTotal, int _nonMaxBlockEdge, int _thresholdSelectedGradient)
{
    thresholdFast = _thresholdFast;
    widthGrid = _widthGrid;
    heightGrid = _heightGrid;
    edgeLeaveOutW = _edgeLeaveOutW;
    edgeLeaveOutH = _edgeLeaveOutH;
    numBlockW = _numBlockW;
    numBlockH = _numBlockH;
    nonMaxBlockEdge = _nonMaxBlockEdge;
    thresholdSelectedGradient = _thresholdSelectedGradient;
    if(nonMaxBlockEdge >= edgeLeaveOutW || nonMaxBlockEdge >= edgeLeaveOutH)
    {
        cout << "[PixelSelector] nonMaxBlockEdge is too large" << endl;
        exit(1);
    }
    for(int i = 0; i < LEVEL; i++)
    {
        numKeyTotal.push_back(_numKeyTotal / (i + 1));
        // numNormalTotal.push_back(_numNormalTotal / (2 * i + 1));
    }
    numNormalTotal.push_back(_numNormalTotal * 0.3);
    numNormalTotal.push_back(_numNormalTotal * 0.3);
    numNormalTotal.push_back(_numNormalTotal * 0.3);
    numNormalTotal.push_back(_numNormalTotal * 0.1);
}

void PixelSelector::selectKeyPointFromImage(shared_ptr<Frame> &frame)
{
    shared_ptr<ImageInfo> ptr_image = frame->image;
    for(int i = 0; i < LEVEL; i++)
    {
        shared_ptr<cv::Mat> ptr_mat = make_shared<cv::Mat>(ptr_image->pyramids.at(i));
        vector<cv::KeyPoint> keyPointsLevel;
        cv::Mat mask = cv::Mat::zeros(ptr_mat->size(), CV_8U);
        int gridW = floor((ptr_mat->cols - 2 * edgeLeaveOutW) / numBlockW);
        int gridH = floor((ptr_mat->rows - 2 * edgeLeaveOutH) / numBlockH);
        int gridNum = ceil(numKeyTotal.at(i) / (numBlockH * numBlockW));
        for(int row = 0; row < numBlockH; row++)
        {
            for(int col = 0; col < numBlockW; col++)
            {
                int x0 = edgeLeaveOutW + col * gridW;
                int x1 = edgeLeaveOutW + (col + 1) * gridW;
                int y0 = edgeLeaveOutH + row * gridH;
                int y1 = edgeLeaveOutH + (row + 1) * gridH;
                x0 = max(x0, edgeLeaveOutW);
                y0 = max(y0, edgeLeaveOutH);
                x1 = min(x1, ptr_mat->cols - edgeLeaveOutW);
                y1 = min(y1, ptr_mat->rows - edgeLeaveOutH);
                cv::Rect gridRect(x0 - 3, y0 - 3, x1 - x0 + 6, y1 - y0 + 6);
                cv::Mat gridROI;
                (*ptr_mat)(gridRect).convertTo(gridROI, CV_8U);
                
                vector<cv::KeyPoint> keyPointsGrid;
                
                int threshold = thresholdFast;
                int maxIteration = 1;
                int countPoints = 0;
                for(int k = 0; k < maxIteration; k++)
                {
                    cv::FAST(gridROI, keyPointsGrid, threshold, true);
                    countPoints = keyPointsGrid.size();
                    if(countPoints - gridNum > (int)(gridNum * 0.1 + 0.5))
                    {
                        threshold += 4;
                    }
                    else if(gridNum - countPoints > (int)(gridNum * 0.1 + 0.5))
                    {
                        threshold -= 16;
                    }
                    else
                    {
                        break;
                    }
                    if(threshold <= 4 || threshold >= 150)
                    {
                        break;
                    }
                }
                for(auto iter = keyPointsGrid.begin(); iter != keyPointsGrid.end();)
                {
                    int x = x0 - 3 + iter->pt.x;
                    int y = y0 - 3 + iter->pt.y;
                    if(x < edgeLeaveOutW || x > ptr_mat->cols - edgeLeaveOutW || y < edgeLeaveOutH || y > ptr_mat->rows - edgeLeaveOutH)
                    {
                        cout << "erase!" << endl;
                        keyPointsGrid.erase(iter);
                    }
                    else
                    {
                        iter->pt.x = x;
                        iter->pt.y = y;
                        keyPointsLevel.push_back(*iter);
                        mask.at<uchar>(y, x) = 1;
                        iter++;
                    }
                }
            }
        }
        frame->keyPointsMask.push_back(mask);
        frame->keyPoints.push_back(keyPointsLevel);
    }
}


void PixelSelector::selectNormalPointFromImage(shared_ptr<Frame> &frame)
{
    shared_ptr<ImageInfo> ptr_image = frame->image;
    for(int i = 0; i < LEVEL; i++)
    {
        shared_ptr<cv::Mat> ptr_mat = make_shared<cv::Mat>(ptr_image->pyramids.at(i));
        shared_ptr<cv::Mat> ptr_dx = make_shared<cv::Mat>(ptr_image->dx.at(i));
        shared_ptr<cv::Mat> ptr_dy = make_shared<cv::Mat>(ptr_image->dy.at(i));
        shared_ptr<cv::Mat> ptr_gradient = make_shared<cv::Mat>(ptr_image->grad_2.at(i));

        vector<cv::Point> normalPointsLevel;
        cv::Mat mask = cv::Mat::zeros(ptr_mat->size(), CV_8U);
        int gridW = floor((ptr_mat->cols - 2 * edgeLeaveOutW) / numBlockW);
        int gridH = floor((ptr_mat->rows - 2 * edgeLeaveOutH) / numBlockH);

        cv::Mat complexity = cv::Mat::zeros(numBlockH, numBlockW, CV_32F);
        cv::Mat distribution = cv::Mat::zeros(numBlockH, numBlockW, CV_32S);


        for(int row = 0; row < numBlockH; row++)
        {
            for(int col = 0; col < numBlockW; col++)
            {
                int x0 = edgeLeaveOutW + col * gridW;
                int x1 = edgeLeaveOutW + (col + 1) * gridW;
                int y0 = edgeLeaveOutH + row * gridH;
                int y1 = edgeLeaveOutH + (row + 1) * gridH;
                x0 = max(x0, edgeLeaveOutW);
                y0 = max(y0, edgeLeaveOutH);
                x1 = min(x1, ptr_mat->cols - edgeLeaveOutW);
                y1 = min(y1, ptr_mat->rows - edgeLeaveOutH);
                cv::Rect gridRect(x0, y0, x1 - x0, y1 - y0);
                cv::Mat gridROIGradient = (*ptr_gradient)(gridRect);
                complexity.at<float>(row, col) = cv::sum(gridROIGradient)[0];
            }
        }
        float totalComplexity = cv::sum(complexity)[0];
        for(int row = 0; row < numBlockH; row++)
        {
            for(int col = 0; col < numBlockW; col++)
            {
                float percentage = complexity.at<float>(row, col) / totalComplexity;
                distribution.at<int>(row, col) = static_cast<int>(numNormalTotal.at(i) * percentage);
                // if(i == LEVEL - 1)
                // {
                //     cout << "row: " << row << " col: " << col << " num: " << distribution.at<int>(row, col) << endl;
                // }
            }
        }
        for(int row = 0; row < numBlockH; row++)
        {
            for(int col = 0; col < numBlockW; col++)
            {
                if(distribution.at<int>(row, col) < 2)
                {
                    continue;
                }
                int x0 = edgeLeaveOutW + col * gridW;
                int x1 = edgeLeaveOutW + (col + 1) * gridW;
                int y0 = edgeLeaveOutH + row * gridH;
                int y1 = edgeLeaveOutH + (row + 1) * gridH;
                x0 = max(x0, edgeLeaveOutW);
                y0 = max(y0, edgeLeaveOutH);
                x1 = min(x1, ptr_mat->cols - edgeLeaveOutW);
                y1 = min(y1, ptr_mat->rows - edgeLeaveOutH);
                cv::Rect gridRect(x0, y0, x1 - x0, y1 - y0);
                cv::Mat gridROIDx = (*ptr_dx)(gridRect);
                cv::Mat gridROIDy = (*ptr_dy)(gridRect);
                cv::Mat gridROIGradient = (*ptr_gradient)(gridRect);
                
                vector<pair<float, cv::Point>> normalPointsGrid;
                for(int y = 0; y < gridROIGradient.rows; y++)
                {
                    for(int x = 0; x < gridROIGradient.cols; x++)
                    {
                        // float val_weight = 0.9 * gridROIDx.at<float>(y, x) + 0.1 * gridROIDy.at<float>(y, x);
                        float val = gridROIGradient.at<float>(y, x);
                        bool isMaxInRegion = true;
                        for(int y_region = -nonMaxBlockEdge; y_region <= nonMaxBlockEdge; y_region++)
                        {
                            for(int x_region = -nonMaxBlockEdge; x_region <= nonMaxBlockEdge; x_region++)
                            {
                                if(gridROIGradient.at<float>(y + y_region, x + x_region) > val)
                                {
                                    isMaxInRegion = false;
                                    break;
                                }
                            }
                            if(!isMaxInRegion)
                            {
                                break;
                            }
                        }
                        if(!isMaxInRegion || val < thresholdSelectedGradient)
                        {
                            continue;
                        }
                        if(val < thresholdSelectedGradient)
                        {
                            continue;
                        }
                        else
                        {
                            normalPointsGrid.emplace_back(val, cv::Point(x0 + x, y0 + y));
                        }
                    }
                }
                sort(normalPointsGrid.begin(), normalPointsGrid.end(), [](const pair<float, cv::Point> &a, const pair<float, cv::Point> &b)
                {
                    return a.first > b.first;
                });
                int numGrid = min((int)normalPointsGrid.size(), distribution.at<int>(row, col));
                for(int k = 0; k < numGrid; k++)
                {
                    normalPointsLevel.push_back(normalPointsGrid.at(k).second);
                    mask.at<uchar>(normalPointsGrid.at(k).second) = 1;
                }
            }
        }
        frame->normalPointsMask.push_back(mask);
        frame->normalPoints.push_back(normalPointsLevel);
    }
}


void PixelSelector::lookKeyPoint(shared_ptr<Frame> &frame)
{
    for(int i = 0; i < LEVEL; i++)
    {
        cv::Mat mat_display;
        frame->image->pyramids.at(i).convertTo(mat_display, CV_8U);
        cv::cvtColor(mat_display, mat_display, cv::COLOR_GRAY2BGR);
        // for(auto iter = frame->keyPoints.at(i).begin(); iter != frame->keyPoints.at(i).end(); iter++)
        // {
        //     cv::circle(mat_display, cv::Point(iter->pt.x, iter->pt.y), 2, cv::Scalar(0, 255, 0), -1);
        // }
        for(int y = 0; y < frame->keyPointsMask.at(i).rows; y++)
        {
            for(int x = 0; x < frame->keyPointsMask.at(i).cols; x++)
            {
                if(frame->keyPointsMask.at(i).at<uchar>(y, x) == 1)
                {
                    cv::circle(mat_display, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
                }
            }
        }
        cv::imshow("pic" + to_string(i), mat_display);
    }
    cv::waitKey(0);
}

void PixelSelector::lookNormalPoint(shared_ptr<Frame> &frame)
{
    for(int i = 0; i < LEVEL; i++)
    {
        cv::Mat mat_display;
        frame->image->pyramids.at(i).convertTo(mat_display, CV_8U);
        cv::cvtColor(mat_display, mat_display, cv::COLOR_GRAY2BGR);
        // for(auto iter = frame->keyPoints.at(i).begin(); iter != frame->keyPoints.at(i).end(); iter++)
        // {
        //     cv::circle(mat_display, cv::Point(iter->pt.x, iter->pt.y), 2, cv::Scalar(0, 255, 0), -1);
        // }
        int total = 0;
        for(int y = 0; y < frame->normalPointsMask.at(i).rows; y++)
        {
            for(int x = 0; x < frame->normalPointsMask.at(i).cols; x++)
            {
                if(frame->normalPointsMask.at(i).at<uchar>(y, x) == 1)
                {
                    cv::circle(mat_display, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), -1);
                    total++;
                }
            }
        }
        cout << "total: " << total << endl;
        cv::imshow("pic" + to_string(i), mat_display);
    }
    cv::waitKey(0);
}