#include "nanoflann.hpp"
#include <pcl-1.14/pcl/point_cloud.h>
#include <pcl-1.14/pcl/visualization/pcl_visualizer.h>
#include <pcl/cloud_iterator.h>
#include <iostream>
#include "point.h"
#include "process_image.h"
#include "frame.h"
#include "pixel_select.h"
#include "tracker.h"
using namespace nanoflann;
using namespace std;
using PointType = pcl::PointXYZ;
using PointCloud = pcl::PointCloud<PointType>;

struct PointClouds
{
    vector<shared_ptr<Point>> points;
    PointClouds(vector<shared_ptr<Point>> _points) : points(_points){}
    inline size_t kdtree_get_point_count() const
    {
        // cout << "hi" << endl;
        return points.size();
    }
    inline float kdtree_get_pt(const size_t index, const size_t dim) const
    {
        if (index >= points.size())
        {
            throw std::out_of_range("index_p2 out of range!");
        }
        if(dim == 0)
        {
            return (float)(points.at(index)->getPositionX());
        }
        else
        {
            return (float)(points.at(index)->getPositionY());
        }
    }
    inline float kdtree_distance(const float *p1, const size_t index_p2, size_t ) const
    {
        if (index_p2 >= points.size())
        {
            throw std::out_of_range("index_p2 out of range!");
        }
        cout << "hi" << endl;
        const auto& p2 = points.at(index_p2);
        cout << "hello" << endl;
        const float d1 = p1[0] - points.at(index_p2)->positionX;
        const float d2 = p1[1] - points.at(index_p2)->positionY;
        return d1 * d1 + d2 * d2;
    }
    template<class BBOX>
    bool kdtree_get_bbox(BBOX& ) const
    {
        return false;
    }
};


int main()
{
    shared_ptr<ImageFolder> folder = make_shared<ImageFolder>("./", "../calibration/kitti.xml");
    folder->readImageFolder();

    shared_ptr<Frame> ref_frame = make_shared<Frame>();
    ref_frame->setFrame(folder->getIndice(0));

    shared_ptr<Frame> tar_frame = make_shared<Frame>();
    tar_frame->setFrame(folder->getIndice(1));

    shared_ptr<PixelSelector> selector = make_shared<PixelSelector>();
    selector->setParameters(40, 20, 10, 5, 5, 20, 10, 2000, 10000, 1, 10);
    selector->selectKeyPointFromImage(ref_frame);

    selector->selectNormalPointFromImage(ref_frame);

    ref_frame->transformToConcern();

    vector<shared_ptr<Point>> tmp = ref_frame->concernNormalPoints.at(0);
    cout << tmp.size() << endl;

    // vector<array<float, 3>> layer_prev = {{10.5f, 20.5f, 10.0f}, {15.2f, 30.1f, 5.0f}, {40.0f, 50.0f, 7.0f}, {80.0f, 90.0f, 8.0f}};
    // vector<array<float, 3>> layer_curr = {{10.0f, 20.0f, 8.0f}, {16.0f, 31.0f, 7.0f}, {39.0f, 49.0f, 8.0f}};
    PointClouds pc(tmp);
    typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointClouds>, PointClouds, 2> KDTree1;
    
    vector<PointClouds> pcs;
    // KDTree1* anotherTrees[2];
    vector<shared_ptr<KDTree1>> anotherTrees;
    for(int i = 0; i < LEVEL; i++)
    {
        pcs.emplace_back(ref_frame->concernNormalPoints.at(i));
    }
    // vector<shared_ptr<KDTree1>> trees;
    
    KDTree1* trees1 = new KDTree1(2, pcs[0], nanoflann::KDTreeSingleIndexAdaptorParams(10));
    KDTree1* trees2 = new KDTree1(2, pcs[1], nanoflann::KDTreeSingleIndexAdaptorParams(10));
    auto tmp1 = make_shared<KDTree1>(2, pcs[0], nanoflann::KDTreeSingleIndexAdaptorParams(10));
    auto tmp2 = make_shared<KDTree1>(2, pcs[1], nanoflann::KDTreeSingleIndexAdaptorParams(10));
    anotherTrees.push_back(tmp1);
    anotherTrees.push_back(tmp2);
    trees1->buildIndex();
    trees2->buildIndex();
    anotherTrees[0]->buildIndex();
    anotherTrees[1]->buildIndex();
    // for(int i = 0; i < LEVEL; i++)
    // {
    //     pcs.emplace_back(ref_frame->concernNormalPoints.at(i));
    //     // auto t = make_shared<KDTree1>(2, pcs[i], nanoflann::KDTreeSingleIndexAdaptorParams(10));
    //     // trees.push_back(t);
    //     // trees.back()->buildIndex();
    //     trees[i] = new KDTree1(2, pcs[i], nanoflann::KDTreeSingleIndexAdaptorParams(10));
    //     trees[i]->buildIndex();
    //     cout << pcs.at(i).kdtree_get_point_count() << endl;
    // }

    for(int i = 0; i < LEVEL; i++)
    {
        for(const auto &point: ref_frame->concernNormalPoints.at(i))
        {
            float query[2] = {point->getPositionX(), point->getPositionY()};
            uint32_t nearest_in;
            float distance;
            if(i == 0)
            {
                anotherTrees[0]->knnSearch(query, 1, &nearest_in, &distance);
            }
            else if(i == 1)
            {
                anotherTrees[1]->knnSearch(query, 1, &nearest_in, &distance);
            }
            else
            {}
            // nearest_index.push_back(static_cast<int>(nearest_in));
            // nearest_distance.push_back(distance);

            // std::cout << "Current Point: (" << point->getPositionX() << ", " << point->getPositionY() << ") -> "
            //         << "Nearest in prev layer: ("
            //         << tmp.at(nearest_in)->getPositionX() << ", "
            //         << tmp.at(nearest_in)->getPositionY() << ")"  << std::endl;
        }
    }
    // for(int i = 0; i < LEVEL; i++)
    // {
    //     delete trees[i];
    // }
    
    // KDTree1 kdtree(2, pc, KDTreeSingleIndexAdaptorParams(10));
    // kdtree.buildIndex();
    // vector<int> nearest_index;
    // vector<float> nearest_distance;
    // for(const auto& point: tmp)
    // {
    //     float query[2] = {point->getPositionX(), point->getPositionY()};
    //     uint32_t nearest_in;
    //     float distance;
    //     kdtree.knnSearch(query, 1, &nearest_in, &distance);
    //     // nearest_index.push_back(static_cast<int>(nearest_in));
    //     // nearest_distance.push_back(distance);

    //     std::cout << "Current Point: (" << point->getPositionX() << ", " << point->getPositionY() << ") -> "
    //               << "Nearest in prev layer: ("
    //               << tmp.at(nearest_in)->getPositionX() << ", "
    //               << tmp.at(nearest_in)->getPositionY() << ")"  << std::endl;
    // }
    // PointCloud::Ptr pcl_layer_prev(new PointCloud());
    // PointCloud::Ptr pcl_layer_curr(new PointCloud());
    // for (const auto& pt : layer_prev)
    //     pcl_layer_prev->points.emplace_back(pt[0], pt[1], pt[2]);
    // for (const auto& pt : layer_curr)
    //     pcl_layer_curr->points.emplace_back(pt[0], pt[1], pt[2]);

    // // 使用 PCL 可视化
    // pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    // viewer->setBackgroundColor(0, 0, 0);

    // // 显示上一层点云（红色）
    // pcl::visualization::PointCloudColorHandlerCustom<PointType> red(pcl_layer_prev, 255, 0, 0);
    // viewer->addPointCloud<PointType>(pcl_layer_prev, red, "layer_prev");
    // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "layer_prev");

    // // 显示当前层点云（绿色）
    // pcl::visualization::PointCloudColorHandlerCustom<PointType> green(pcl_layer_curr, 0, 255, 0);
    // viewer->addPointCloud<PointType>(pcl_layer_curr, green, "layer_curr");
    // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "layer_curr");

    // // 显示父子关系（线条连接）
    // for (size_t i = 0; i < layer_curr.size(); ++i) {
    //     const auto& child = pcl_layer_curr->points[i];
    //     const auto& parent = pcl_layer_prev->points[nearest_index[i]];
    //     viewer->addLine<PointType, PointType>(child, parent, "line_" + std::to_string(i));
    // }

    // // 启动可视化器
    // viewer->addCoordinateSystem(1.0);
    // viewer->initCameraParameters();
    // while (!viewer->wasStopped()) {
    //     viewer->spinOnce(100);
    // }

    return 0;
}