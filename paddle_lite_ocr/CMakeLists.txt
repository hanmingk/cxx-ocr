cmake_minimum_required(VERSION 3.14)
project(ppocr_lite VERSION 0.1.0 LANGUAGES C CXX)

SET(PADDLE_LITE_LIB "/home/hmk/third_partylib/paddle_lite_inference" CACHE PATH "Location of libraries")
SET(OPENCV_DIR "/home/hmk/third_partylib/opencv-4.9.0" CACHE PATH "Location of libraries")

if(NOT DEFINED PADDLE_LITE_LIB)
    message(FATAL_ERROR "please set PADDLE_LIB with -DPADDLE_LIB=/path/paddle/lib")
endif()

if(NOT DEFINED OPENCV_DIR)
    message(FATAL_ERROR "please set OPENCV_DIR with -DOPENCV_DIR=/path/opencv")
endif()

if (WIN32)
    include_directories(${PADDLE_LITE_LIB}/cxx/include)
    link_directories(${PADDLE_LITE_LIB}/cxx/lib)
    find_package(OpenCV REQUIRED PATHS ${OPENCV_DIR}/build/ NO_DEFAULT_PATH)

else ()
    find_package(OpenCV REQUIRED PATHS ${OPENCV_DIR}/lib64/cmake/ NO_DEFAULT_PATH)
    include_directories(${PADDLE_LITE_LIB}/cxx/include)
    link_directories(${PADDLE_LITE_LIB}/cxx/lib)
    link_directories(${PADDLE_LITE_LIB}/third_party/mklml/lib)
    include_directories(${PADDLE_LITE_LIB}/third_party/mklml/include)
endif ()

include_directories(${OpenCV_INCLUDE_DIRS})

# set(DEPS ${DEPS} ${OpenCV_LIBS})
set(DEPS opencv_core opencv_imgcodecs opencv_imgproc)
# set(DEPS ${DEPS} paddle_full_api_shared iomp5)
set(DEPS ${DEPS} paddle_full_api_shared iomp5)

# 设置编译选项，启用 AVX
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -pthread -mavx")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -pthread -mavx")

include_directories("${CMAKE_SOURCE_DIR}/")
file(GLOB SRCS "src/*.cpp")

add_executable(${PROJECT_NAME} ${SRCS})
target_link_libraries(${PROJECT_NAME} ${DEPS})
