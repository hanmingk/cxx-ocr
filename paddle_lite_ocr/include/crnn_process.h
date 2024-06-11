#pragma once

#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "math.h" //NOLINT
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

cv::Mat CrnnResizeImg(cv::Mat img, float wh_ratio, int rec_image_height);

std::vector<std::string> ReadDict(const std::string &path);

cv::Mat GetRotateCropImage(cv::Mat img, const std::vector<int> &box);

template <class ForwardIterator>
inline size_t Argmax(ForwardIterator first, ForwardIterator last) {
  return std::distance(first, std::max_element(first, last));
}
