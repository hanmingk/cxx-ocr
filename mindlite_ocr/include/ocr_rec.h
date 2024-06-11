#pragma once

#include "opencv2/core.hpp"

#include "include/ocr_api.h"
#include "include/ocr_model.h"

class CRNNRecognize : public MDOCRModel {
  public:
    explicit CRNNRecognize(const Flags &flags);
    ~CRNNRecognize() = default;

    mindspore::Status RunPredict(const cv::Mat &img, std::string &rec_text,
                                 float &rec_text_score);
    static cv::Mat CrnnResizeImg(const cv::Mat &img, float wh_ratio,
                                 int rec_img_height);
    static std::vector<std::string> ReadDict(const std::string &dict_path);

  private:
    mindspore::Status SetInputData(const float *data, size_t data_size);

    std::vector<std::string> charactor_dict;
    int rec_image_height;
};