#顶层目录设定cmake版本
cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_FLAGS "-fsanitize=address -g")
#设定工程名字
project(main)
#索引工程子目录
add_subdirectory(hog_feature)
add_subdirectory(mysvm)

set(PROJECT_BINARY_DIR ${PROJECT_SOURCE_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

add_executable(main main.cpp)

#opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(main ${OpenCV_LIBS} libhog libsvm)
