cmake_minimum_required(VERSION 3.6)
project(framework)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-std=c++17 -O3")

include_directories(${PROJECT_SOURCE_DIR})

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(fmt REQUIRED)
find_package(PCL REQUIRED)

file(GLOB SOURCES
    process_image.cpp
    global_params.cpp
    frame.cpp
    pixel_select.cpp
    point.cpp
    light_affine.cpp
    tracker.cpp
    trackerDSO.cpp
    tracker2ml.cpp
    trajectory.cpp
    immature_point.cpp
    map.cpp
    map_point.cpp
)

include_directories(${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} "/usr/local/include/fmt")

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(framework main.cpp ${SOURCES})
add_executable(debug debug.cpp)
# add_executable(debug2 debug2.cpp ${SOURCES})
add_executable(debug3 debug3.cpp)

target_link_libraries(framework ${OpenCV_LIBS} OpenMP::OpenMP_CXX fmt::fmt ${PCL_LIBRARIES})
target_link_libraries(debug ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
# target_link_libraries(debug2 ${OpenCV_LIBS} OpenMP::OpenMP_CXX fmt::fmt ${PCL_LIBRARIES})
target_link_libraries(debug3 ${OpenCV_LIBS} OpenMP::OpenMP_CXX fmt::fmt ${PCL_LIBRARIES})