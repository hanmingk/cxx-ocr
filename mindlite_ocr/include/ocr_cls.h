#pragma once

#include "include/ocr_model.h"
#include "opencv2/core.hpp"

class Classifier : public MDOCRModel {
  public:
    explicit Classifier(double cls_thresh) : cls_thresh_(cls_thresh){};
    ~Classifier() = default;

    mindspore::Status RunPredict(const cv::Mat &img, bool &is_rotate);
    static cv::Mat ClsResizeImg(const cv::Mat &img);
    inline static cv::Mat ClsRotateImg(const cv::Mat &img) {
        cv::Mat rotate_img;
        cv::rotate(img, rotate_img, 1);
        return rotate_img;
    }

  private:
    mindspore::Status SetInputData(const float *data, size_t data_size);

    double cls_thresh_ = 0.9;
};