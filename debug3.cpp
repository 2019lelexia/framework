#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <iostream>

using namespace std;
using namespace Eigen;

int main() {
    // 模拟一组SE3位姿数据（平移部分）
    vector<Isometry3d> poses;
    for (int i = 0; i < 10; ++i) {
        Isometry3d pose = Isometry3d::Identity();
        pose.translate(Vector3d(i * 1.0, i * 2.0, i * 3.0));  // 随机的平移
        poses.push_back(pose);
    }

    // 创建 PCL 可视化器
    pcl::visualization::PCLVisualizer viewer("Trajectory Viewer");

    // 设置轨迹线条的颜色和宽度
    viewer.setBackgroundColor(0.0, 0.0, 0.0); // 设置背景色为黑色
    viewer.addCoordinateSystem(1.0);           // 添加坐标轴
    viewer.setCameraPosition(0, 0, 10, 0, 0, 0); // 设置视角

    // 绘制轨迹：依次连接位姿的平移部分
    for (size_t i = 1; i < poses.size(); ++i) {
        // 提取连续两帧的平移向量
        Vector3d t1 = poses[i - 1].translation();
        Vector3d t2 = poses[i].translation();

        // 使用 PCL 绘制两点之间的线段
        string line_id = "line_" + to_string(i);
        viewer.addLine<pcl::PointXYZ>(pcl::PointXYZ(t1.x(), t1.y(), t1.z()),
                                      pcl::PointXYZ(t2.x(), t2.y(), t2.z()), line_id);
    }

    // 持续显示
    while (!viewer.wasStopped()) {
        viewer.spinOnce(100);  // 每100毫秒刷新一次视图
    }

    return 0;
}
