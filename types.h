#pragma once

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/StdVector>
#include <sophus/so3.hpp>
#include <sophus/se3.hpp>
#include <sophus/sim3.hpp>


using namespace std;

typedef Sophus::SE3d SE3;
typedef Sophus::SO3d SO3;
typedef Sophus::Sim3d Sim3;

typedef Eigen::Vector2d Vector2d;
typedef Eigen::Vector2f Vector2f;
typedef Eigen::Vector3f Vector3f;
typedef Eigen::Vector3d Vector3d;

typedef Eigen::Matrix<float, 8, 1> Vector8f;
typedef Eigen::Matrix<double, 8, 1> Vector8d;

typedef Eigen::Matrix<double, 3, 3> Matrix3d;
typedef Eigen::Matrix<float, 3, 3> Matrix3f;

typedef Eigen::Matrix<double, 10, 1> Vector10d;
typedef Eigen::Matrix<float, 10, 1> Vector10f;

typedef Eigen::Matrix<float, 8, 8> Matrix8f;
typedef Eigen::Matrix<double, 8, 8> Matrix8d;
