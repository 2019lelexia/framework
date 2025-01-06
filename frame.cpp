#include "frame.h"

Frame::Frame()
{
    for(int i = 0; i < LEVEL; i++)
    {
        numPoints.push_back(0);
    }
}

Frame::~Frame()
{}

inline float myPointCloudFlann::kdtree_get_pt(const size_t index, const size_t dim) const
{
    if(dim == 0)
    {
        return (float)(points.at(index)->getPositionX());
    }
    else
    {
        return (float)(points.at(index)->getPositionY());
    }
}

inline float myPointCloudFlann::kdtree_distance(const float *p1, const size_t index_p2, size_t ) const
{
    const auto& p2 = points.at(index_p2);
    const float d1 = p1[0] - points.at(index_p2)->positionX;
    const float d2 = p1[1] - points.at(index_p2)->positionY;
    return d1 * d1 + d2 * d2;
}

void Frame::setFrame(shared_ptr<ImageInfo> _image)
{
    image = _image;
    affine = AffineLight();
    exposure = 1.0;
}

void Frame::transformToConcern()
{
    shared_ptr<Frame> ptr_frame = make_shared<Frame>(*this);
    for(int i = 0; i < LEVEL; i++)
    {
        vector<shared_ptr<Point>> pointsLevel;
        for(auto iter = keyPoints.at(i).begin(); iter != keyPoints.at(i).end(); iter++)
        {
            shared_ptr<Point> tmpPoint = make_shared<Point>(*iter, i, ptr_frame);
            pointsLevel.push_back(tmpPoint);
        }
        concernKeyPoints.push_back(pointsLevel);
        numPoints.at(i) += concernKeyPoints.at(i).size();
    }
    for(int i = 0; i < LEVEL; i++)
    {
        vector<shared_ptr<Point>> pointsLevel;
        for(auto iter = normalPoints.at(i).begin(); iter != normalPoints.at(i).end(); iter++)
        {
            shared_ptr<Point> tmpPoint = make_shared<Point>(*iter, i, ptr_frame);
            pointsLevel.push_back(tmpPoint);
        }
        concernNormalPoints.push_back(pointsLevel);
        numPoints.at(i) += concernNormalPoints.at(i).size();
    }
}

Vector3f Frame::interpolationPixelDxDy(int level, float u, float v)
{
    int x1 = (int)(u);
    int y1 = (int)(v);
    int x2 = min(x1 + 1, wG[level] - 1);
    int y2 = min(y1 + 1, hG[level] - 1);
    float dx = u - x1;
    float dy = v - y1;
    float dxdy = dx * dy;

    Vector3f q11(image->pyramids.at(level).at<float>(y1, x1), image->dx.at(level).at<float>(y1, x1), image->dy.at(level).at<float>(y1, x1));
    Vector3f q12(image->pyramids.at(level).at<float>(y1, x2), image->dx.at(level).at<float>(y1, x2), image->dy.at(level).at<float>(y1, x2));
    Vector3f q21(image->pyramids.at(level).at<float>(y2, x1), image->dx.at(level).at<float>(y2, x1), image->dy.at(level).at<float>(y2, x1));
    Vector3f q22(image->pyramids.at(level).at<float>(y2, x2), image->dx.at(level).at<float>(y2, x2), image->dy.at(level).at<float>(y2, x2));
    
    Vector3f r1 = (1 - dx) * q11 + dx * q12;
    Vector3f r2 = (1 - dx) * q21 + dx * q22;
    // Vector3f p = dy * r1 + (1 - y1) * r2;
    Vector3f p = dxdy * q22 + (dx - dxdy) * q12 + (dy - dxdy) * q21 + (1 - dx - dy - dxdy) * q11;

    return p;
}

float Frame::interpolationPixel(int level, float u, float v)
{
    int x1 = (int)(u);
    int y1 = (int)(v);
    int x2 = min(x1 + 1, wG[level] - 1);
    int y2 = min(y1 + 1, hG[level] - 1);
    float dx = u - x1;
    float dy = v - y1;
    float dxdy = dx * dy;

    float q11(image->pyramids.at(level).at<float>(y1, x1));
    float q12(image->pyramids.at(level).at<float>(y1, x2));
    float q21(image->pyramids.at(level).at<float>(y2, x1));
    float q22(image->pyramids.at(level).at<float>(y2, x2));

    float r1 = (x2 - u) * q11 + (u - x1) * q12;
    float r2 = (x2 - u) * q21 + (u - x1) * q22;
    // float p = (y2 - v) * r1 + (v - y1) * r2;
    float p = dxdy * q22 + (dx - dxdy) * q12 + (dy - dxdy) * q21 + (1 - dx - dy - dxdy) * q11;

    return p;
}

void Frame::makePointCloudFlann()
{
    typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, myPointCloudFlann>, myPointCloudFlann, 2> KDTree;
    vector<myPointCloudFlann> pcs;
    // vector<KDTree> trees;
    vector<shared_ptr<KDTree>> trees;
    for(int i = 0; i < LEVEL; i++)
    {
        pcs.emplace_back(concernNormalPoints.at(i).size(), concernNormalPoints.at(i));
        cout << pcs.at(i).kdtree_get_point_count() << endl;
    }
    for(int i = 0; i < LEVEL; i++)
    {
        auto tmp = make_shared<KDTree>(2, pcs[i], nanoflann::KDTreeSingleIndexAdaptorParams(5));
        trees.push_back(tmp);
        trees.back()->buildIndex();
    }
    const int numNeighbour = 5;
    const int numParent = 1;

    for(int level = 0; level < LEVEL; level++)
    {
        nanoflann::KNNResultSet<float, int, int> resultNeighbour(numNeighbour);
        nanoflann::KNNResultSet<float, int, int> resultParent(numParent);
        for(const auto& point: concernNormalPoints.at(level))
        {
            int searchIndex[numNeighbour];
            float searchDistance[numNeighbour];
            vector<int> nnIndex;
            vector<float> nnDistance;
            resultNeighbour.init(searchIndex, searchDistance);
            float coordinate[2] = {point->getPositionX(), point->getPositionY()};
            trees.at(level)->findNeighbors(resultNeighbour, (float *) &coordinate, nanoflann::SearchParameters());
            // trees.at(level)->knnSearch(coordinate, numNeighbour, searchIndex, searchDistance);
            for(int k = 0; k < numNeighbour; k++)
            {
                point->indexNeighbour[k] = static_cast<int>(searchIndex[k]);
                point->distanceNeighbour[k] = searchDistance[k];
                if(searchIndex[k] < 0 || searchIndex[k] >= pcs[level].num)
                {
                    cout << "the index is out of range" << endl;
                    exit(0);
                }
            }
            if(level < LEVEL - 1)
            {
                int searchIndexParent;
                float searchDistanceParent;
                resultParent.init(&searchIndexParent, &searchDistanceParent);
                float coordinateParent[2] = {point->getPositionX() * 0.5f, point->getPositionY() * 0.5f};
                trees.at(level + 1)->findNeighbors(resultParent, (float *) &coordinateParent, nanoflann::SearchParameters());
                // trees.at(level + 1)->knnSearch(coordinateParent, numParent, &searchIndexParent, &searchDistanceParent);
                point->indexParent = static_cast<int>(searchIndexParent);
                point->distanceParent = searchDistanceParent;
                if(searchIndexParent < 0 || searchIndexParent >= pcs[level + 1].num)
                {
                    cout << "the index is out of range" << endl;
                    exit(1);
                }
            }
        }
    }
}

void Frame::visualizePointCloudLevel(int level, const SE3& transform)
{
    shared_ptr<pcl::PointCloud<pcl::PointXYZ>> pointcloud = make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    float fxL = fxG[level];
    float fyL = fyG[level];
    float cxL = cxG[level];
    float cyL = cyG[level];
    for(const auto& point: concernNormalPoints.at(level))
    {
        float d = 1 / point->depthNew;
        float x = (point->getPositionX() - cxL) * d / fxL;
        float y = (point->getPositionY() - cyL) * d / fyL;
        Eigen::Vector4d tmpPoint(x, y, d, 1.0);
        Eigen::Vector4d pointHomogeneous = transform.matrix() * tmpPoint;
        pointcloud->points.emplace_back(pointHomogeneous[0], pointHomogeneous[1], pointHomogeneous[2]);
        // pointcloud->points.emplace_back(x, y, d);
    }
    shared_ptr<pcl::visualization::PCLVisualizer> viewer = make_shared<pcl::visualization::PCLVisualizer>("point cloud viewer");
    viewer->getRenderWindow()->GlobalWarningDisplayOff();
    viewer->setBackgroundColor(0, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> red(pointcloud, 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(pointcloud, red, "redcloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "redcloud");

    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    while(!viewer->wasStopped())
    {
        viewer->spinOnce(100);
    }
}

void Frame::visualizePointCloudAllLevel()
{
    vector<shared_ptr<pcl::PointCloud<pcl::PointXYZ>>> pointcloudVector;
    for(int i = 0; i < LEVEL; i++)
    {
        pointcloudVector.emplace_back(make_shared<pcl::PointCloud<pcl::PointXYZ>>());
    }
    for(int i = 0; i < LEVEL; i++)
    {
        float fxL = fxG[i];
        float fyL = fyG[i];
        float cxL = cxG[i];
        float cyL = cyG[i];
        for(const auto& point: concernNormalPoints.at(i))
        {
            float d = 1 / point->depth;
            float x = (point->getPositionX() - cxL) * d / fxL;
            float y = (point->getPositionY() - cyL) * d / fyL;
            pointcloudVector.at(i)->points.emplace_back(x, y, d);
        }
    }
    shared_ptr<pcl::visualization::PCLVisualizer> viewer = make_shared<pcl::visualization::PCLVisualizer>("point cloud viewer");
    viewer->getRenderWindow()->GlobalWarningDisplayOff();
    viewer->setBackgroundColor(255, 255, 255);
    for(int i = 0; i < LEVEL; i++)
    {
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> green(pointcloudVector.at(i), 0, 255, 0);
        viewer->addPointCloud<pcl::PointXYZ>(pointcloudVector.at(i), green, "greencloud");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "greencloud");
    }
    // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> green(pointcloud, 0, 255, 0);
    // viewer->addPointCloud<pcl::PointXYZ>(pointcloud, green, "greencloud");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    while(!viewer->wasStopped())
    {
        viewer->spinOnce(100);
    }
}