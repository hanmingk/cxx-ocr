#pragma once

#include <memory>

#include "include/ocr_api.h"
#include "include/ocr_cls.h"
#include "include/ocr_det.h"
#include "include/ocr_rec.h"

class MindOCR {
  public:
    explicit MindOCR();
    ~MindOCR() = default;

    mindspore::Status Init(const Flags &flags);

    mindspore::Status DetPredict(const cv::Mat &img,
                                 std::vector<std::vector<int32_t>> &boxes);

    mindspore::Status RecPredict(const cv::Mat &img, std::string &rec_text,
                                 float &rec_text_score, bool cls);

  private:
    std::unique_ptr<DBDetector> detector_;
    std::unique_ptr<CRNNRecognize> recognize_;
    std::unique_ptr<Classifier> classifier_;
};