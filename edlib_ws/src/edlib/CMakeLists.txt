cmake_minimum_required(VERSION 3.0.1)
project(edlib)

list(APPEND CMAKE_PREFIX_PATH "/usr/local/lib")
set(OpenCV_DIR /usr/lib/x86_64-linux-gnu/cmake/opencv4)
find_package(catkin REQUIRED COMPONENTS
  OpenCV
  roscpp
  rospy
  std_msgs
  geometry_msgs
  image_transport
  camera_info_manager
  cv_bridge
  sensor_msgs

)
find_package(OpenCV REQUIRED)
add_compile_options(-std=c++11)
catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  include
  /opt/MVS/include/
  /usr/include/eigen3
)
add_library(EDLib STATIC
        src/ED.cpp
        include/ED.h
        src/EDCircles.cpp
        include/EDCircles.h
        src/EDColor.cpp
        include/EDColor.h
        include/EDLib.h
        src/EDLines.cpp
        include/EDLines.h
        src/EDPF.cpp
        include/EDPF.h
        src/NFA.cpp
        include/NFA.h
)
LINK_DIRECTORIES(
/opt/MVS/lib/64/
)

add_executable(video_test22 src/video_test22.cpp)

add_dependencies(video_test22 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(EDLib ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_executable(video_test3 src/video_test3.cpp)
target_link_libraries(video_test3
  ${catkin_LIBRARIES}
)
target_link_libraries(video_test3 ${catkin_LIBRARIES})

target_link_libraries(EDLib
  ${catkin_LIBRARIES}
)

target_link_libraries(video_test22
 EDLib
 ${catkin_LIBRARIES}
 ${OpenCV_LIBRARIES}
)
