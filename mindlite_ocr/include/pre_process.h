#pragma once

#include <vector>

#include "opencv2/core.hpp"

void AvxMeanScale(const float *din, float *dout, size_t size,
                  const std::vector<float> &mean,
                  const std::vector<float> &scale);

cv::Mat GetRotateCropImage(const cv::Mat &img, const std::vector<int> &box);