cmake_minimum_required(VERSION 3.10)
project(FaceDetectionBlurrCpp)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/functions.cpp
    src/main.cpp
)

add_executable(FaceDetectionBlurrCpp ${SOURCES})

target_link_libraries(FaceDetectionBlurrCpp ${OpenCV_LIBS})
