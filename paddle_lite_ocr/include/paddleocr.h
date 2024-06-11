#pragma once

#include <include/ocr_det.h>
#include <include/ocr_rec.h>
#include <include/utils.h>

namespace PaddleOCR
{
    class PPOCR
    {
    public:
        explicit PPOCR(const Flags &flags);
        ~PPOCR() = default;

        OCRPredictResultArray ocr(const cv::Mat img, bool det = true, bool rec = true);

    private:
        std::unique_ptr<DBDetector> detector_;
        std::unique_ptr<CRNNRecognizer> recognizer_;

        std::vector<std::string> charactor_dict;
    };
}