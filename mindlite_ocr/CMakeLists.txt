cmake_minimum_required(VERSION 3.14)
project(md_ocr VERSION 0.1.0 LANGUAGES C CXX)

set(MINDSPORE_LITE_DIR "/home/hmk/third_partylib/mindspore-lite-2.2.14" CACHE PATH "Location of mindspore lite libraries")
set(OPENCV_DIR "/home/hmk/third_partylib/opencv-4.9.0" CACHE PATH "Location of opencv libraries")

if (NOT DEFINED MINDSPORE_LITE_DIR)
    message(FATAL_ERROR "Please set MINDSPORE_LITE_DIR with -DMINDSPOR_LITE_DIR=/path/mindspore")
endif()

if (NOT DEFINED OPENCV_DIR)
    message(FATAL_ERROR "Please set OPENCV_DIR with -DOPENCV_DIR=/path/opencv")
endif()

find_package(OpenCV REQUIRED PATHS ${OPENCV_DIR}/lib64/cmake/opencv4 NO_DEFAULT_PATH)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${MINDSPORE_LITE_DIR}/runtime)
include_directories(${MINDSPORE_LITE_DIR}/runtime/include)

link_directories(${MINDSPORE_LITE_DIR}/runtime/lib)
link_directories(${MINDSPORE_LITE_DIR}/runtime/third_party/glog)

set(DEPS opencv_core opencv_imgcodecs opencv_imgproc)
set(DEPS ${DEPS} mindspore-lite pthread dl)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -mavx")

include_directories(${CMAKE_SOURCE_DIR}/)
file(GLOB SRCS ${CMAKE_SOURCE_DIR}/src/*.cc)

add_executable(${PROJECT_NAME} ${SRCS})
target_link_libraries(${PROJECT_NAME} ${DEPS})

