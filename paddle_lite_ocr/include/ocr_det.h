#pragma once

#include <include/utils.h>
#include <include/flags.h>

using namespace paddle;

namespace PaddleOCR
{
    // resize image to a size multiple of 32 which is required by the network
    cv::Mat DetResizeImg(const cv::Mat img, int max_size_len, std::vector<float> &ratio_hw);

    class DBDetector
    {
    public:
        explicit DBDetector(const Flags &flags);
        ~DBDetector() = default;

        // Run predictor
        void Run(const cv::Mat img, std::vector<std::vector<int>> &boxes);

    private:
        std::shared_ptr<lite_api::PaddlePredictor> predictor_;

        int max_side_len;
        double det_db_thresh;
        int det_db_use_dilate;
        float det_db_box_thresh;
        float det_db_unclip_ratio;
        int det_use_polygon_score;
    };
}