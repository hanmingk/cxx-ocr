#pragma once

#include <include/utils.h>
#include <include/flags.h>

using namespace paddle;

namespace PaddleOCR
{
    class CRNNRecognizer
    {
    public:
        explicit CRNNRecognizer(const Flags &flags);
        ~CRNNRecognizer() = default;

        void Run(const cv::Mat img, const std::vector<std::vector<int>> &box,
                 const std::vector<std::string> &charactor_dict,
                 std::vector<std::string> &rec_text, std::vector<float> &rec_text_score);

        cv::Mat RunClsModel(const cv::Mat img, const float thresh = 0.9);

    private:
        std::shared_ptr<lite_api::PaddlePredictor> rec_predictor_;
        std::shared_ptr<lite_api::PaddlePredictor> cls_predictor_;

        int use_direction_classify;
        int rec_image_height;
    };
}