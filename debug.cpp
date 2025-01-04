#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

// 计算每个格子的梯度总和（复杂度）
void computeGridComplexity(const cv::Mat& grad_magnitude, int grid_size, cv::Mat& complexity_map) {
    complexity_map = cv::Mat::zeros((grad_magnitude.rows + grid_size - 1) / grid_size, 
                                    (grad_magnitude.cols + grid_size - 1) / grid_size, CV_32F);
    for (int grid_y = 0; grid_y < grad_magnitude.rows; grid_y += grid_size) {
        for (int grid_x = 0; grid_x < grad_magnitude.cols; grid_x += grid_size) {
            cv::Rect grid_rect(grid_x, grid_y, grid_size, grid_size);
            grid_rect &= cv::Rect(0, 0, grad_magnitude.cols, grad_magnitude.rows); // 边界裁剪
            cv::Mat grid = grad_magnitude(grid_rect);
            float sum = cv::sum(grid)[0]; // 梯度强度总和
            complexity_map.at<float>(grid_y / grid_size, grid_x / grid_size) = sum;
        }
    }
}

// 根据复杂度动态分配选点数量
void distributePoints(const cv::Mat& complexity_map, int total_points, cv::Mat& points_map) {
    float total_complexity = cv::sum(complexity_map)[0];
    points_map = cv::Mat::zeros(complexity_map.size(), CV_32S);
    int total = 0;

    for (int y = 0; y < complexity_map.rows; ++y) {
        for (int x = 0; x < complexity_map.cols; ++x) {
            float complexity = complexity_map.at<float>(y, x);
            int points = static_cast<int>((complexity / total_complexity) * total_points);
            points_map.at<int>(y, x) = points;
            total += points;
        }
    }
    std::cout << "total distribute: " << total << std::endl;
}

void nonMaxSuppression(const cv::Mat& src, cv::Mat& dst, int block_size) {
    cv::Mat max_mask = cv::Mat::zeros(src.size(), CV_8U); // 创建掩膜
    dst = cv::Mat::zeros(src.size(), CV_32F);

    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            // 当前像素值
            float current_value = src.at<float>(y, x);

            // 查找局部最大值
            bool is_max = true;
            for (int dy = -block_size; dy <= block_size; ++dy) {
                for (int dx = -block_size; dx <= block_size; ++dx) {
                    int ny = y + dy, nx = x + dx;
                    if (nx >= 0 && ny >= 0 && nx < src.cols && ny < src.rows) {
                        if (src.at<float>(ny, nx) > current_value) {
                            is_max = false;
                            break;
                        }
                    }
                }
                if (!is_max) break;
            }

            if (is_max) {
                max_mask.at<uchar>(y, x) = 1; // 标记为局部最大
                dst.at<float>(y, x) = current_value;
            }
        }
    }
}

// 栅格内选点
void selectPointsInGrid(const cv::Mat& grid, const cv::Point& grid_offset, int num_points, cv::Mat& result) {
    cv::Mat suppressed_grid;
    nonMaxSuppression(grid, suppressed_grid, 1); // 使用 block_size = 1 进行非极大值抑制

    std::vector<std::pair<float, cv::Point>> candidates;
    for (int y = 0; y < grid.rows; ++y) {
        for (int x = 0; x < grid.cols; ++x) {
            float val = suppressed_grid.at<float>(y, x);
            if (val > 0) { // 只保留抑制后的非零点
                candidates.emplace_back(val, cv::Point(x + grid_offset.x, y + grid_offset.y));
            }
        }
    }

    // 根据梯度强度排序并选点
    std::sort(candidates.begin(), candidates.end(),
              [](const std::pair<float, cv::Point>& a, const std::pair<float, cv::Point>& b) {
                  return a.first > b.first; // 按梯度值降序排序
              });

    for (int i = 0; i < std::min((int)candidates.size(), num_points); ++i) {
        result.at<uchar>(candidates[i].second) = 255; // 标记为选中
    }
}

// 绘制栅格和分配的点数
void drawGridAndLabels(cv::Mat& display, int grid_size, const cv::Mat& points_map) {
    int grid_rows = points_map.rows, grid_cols = points_map.cols;
    for (int r = 0; r < grid_rows; ++r) {
        for (int c = 0; c < grid_cols; ++c) {
            cv::Point top_left(c * grid_size, r * grid_size);
            cv::Point bottom_right((c + 1) * grid_size - 1, (r + 1) * grid_size - 1);

            // 绘制格子
            cv::rectangle(display, top_left, bottom_right, cv::Scalar(0, 255, 0), 1);

            // 显示分配的点数
            int points = points_map.at<int>(r, c);
            cv::putText(display, std::to_string(points), 
                        cv::Point(top_left.x, top_left.y + grid_size / 3), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
        }
    }
}

// 主函数：基于梯度复杂度的自适应选点
void adaptiveKeypointSelection(const cv::Mat& img, cv::Mat& result, int grid_size, int total_points) {
    // 1. 计算梯度
    cv::Mat grad_x, grad_y, grad_magnitude;
    cv::Sobel(img, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(img, grad_y, CV_32F, 0, 1, 3);
    cv::magnitude(grad_x, grad_y, grad_magnitude);

    // 2. 计算复杂度地图
    cv::Mat complexity_map;
    computeGridComplexity(grad_magnitude, grid_size, complexity_map);

    // 3. 分配每个格子的选点数量
    cv::Mat points_map;
    distributePoints(complexity_map, total_points, points_map);

    // 4. 在每个格子中选点
    result = cv::Mat::zeros(img.size(), CV_8U);
    for (int grid_y = 0; grid_y < grad_magnitude.rows; grid_y += grid_size) {
        for (int grid_x = 0; grid_x < grad_magnitude.cols; grid_x += grid_size) {
            cv::Rect grid_rect(grid_x, grid_y, grid_size, grid_size);
            grid_rect &= cv::Rect(0, 0, grad_magnitude.cols, grad_magnitude.rows); // 保证不超出边界

            int num_points = points_map.at<int>(grid_y / grid_size, grid_x / grid_size);
            if (num_points > 0) {
                cv::Mat grid = grad_magnitude(grid_rect);
                selectPointsInGrid(grid, cv::Point(grid_x, grid_y), num_points, result);
            }
        }
    }

    // 5. 绘制结果
    cv::Mat display;
    cv::cvtColor(img, display, cv::COLOR_GRAY2BGR);
    drawGridAndLabels(display, grid_size, points_map); // 绘制格子和标签
    int total;
    for (int y = 0; y < result.rows; ++y) {
        for (int x = 0; x < result.cols; ++x) {
            if (result.at<uchar>(y, x) == 255) {
                cv::circle(display, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
                total++;
            }
        }
    }
    std::cout << "actually total: " << total << std::endl;

    cv::imshow("Adaptive Keypoints with Grid", display);
    cv::waitKey(0);
}

// 测试程序
int main() {
    // 1. 读取灰度图像
    std::string image_path = "./000001.png"; // 替换为你的图像路径
    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Failed to load image!" << std::endl;
        return -1;
    }

    // 2. 自适应选点
    cv::Mat result;
    adaptiveKeypointSelection(img, result, 50, 10000); // 每格50x50，总共2000个点

    return 0;
}
