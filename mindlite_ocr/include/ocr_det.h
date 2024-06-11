#pragma once

#include "opencv2/core.hpp"

#include "include/ocr_api.h"
#include "include/ocr_model.h"

class DBDetector : public MDOCRModel {
  public:
    explicit DBDetector(const Flags &flags);
    ~DBDetector() = default;

    mindspore::Status RunPredict(const cv::Mat &img,
                                 std::vector<std::vector<int>> &boxes);

    static cv::Mat DetResizeImg(const cv::Mat img, int max_size_len,
                                std::vector<float> &ratio_hw);

  private:
    mindspore::Status SetInputData(const float *data, size_t data_size);

    int det_db_use_dilate;
    float det_db_box_thresh;
    float det_db_unclip_ratio;
    int det_use_polygon_score;
    double det_db_thresh;
};