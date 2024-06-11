#include <include/paddleocr.h>
#include <include/crnn_process.h>

cv::Mat Visualization(cv::Mat srcimg,
                      std::vector<std::vector<int>> boxes)
{
    cv::Point rook_points[boxes.size()][4];
    for (int n = 0; n < boxes.size(); n++)
    {
        for (int m = 0; m < 4; m++)
        {
            rook_points[n][m] = cv::Point(static_cast<int>(boxes[n][m * 2]),
                                          static_cast<int>(boxes[n][m * 2 + 1]));
        }
    }
    cv::Mat img_vis;
    srcimg.copyTo(img_vis);
    for (int n = 0; n < boxes.size(); n++)
    {
        const cv::Point *ppt[1] = {rook_points[n]};
        int npt[] = {4};
        cv::polylines(img_vis, ppt, npt, 1, 1, CV_RGB(0, 255, 0), 2, 8, 0);
    }

    cv::imwrite("../asset/vis.jpg", img_vis);
    std::cout << "The detection visualized image saved in ./vis.jpg" << std::endl;
    return img_vis;
}

namespace PaddleOCR
{
    PPOCR::PPOCR(const Flags &flags)
    {
        this->charactor_dict = ReadDict(flags.dict_path);
        charactor_dict.insert(charactor_dict.begin(), "#"); // blank char for ctc
        charactor_dict.push_back(" ");

        this->detector_.reset(new DBDetector(flags));
        this->recognizer_.reset(new CRNNRecognizer(flags));
    }

    OCRPredictResultArray PPOCR::ocr(const cv::Mat img, bool det, bool rec)
    {
        std::vector<std::vector<int>> boxes;
        if (det)
        {
            this->detector_->Run(img, boxes);
        }

        if (rec && !det)
        {
            int width = img.cols;
            int height = img.rows;
            boxes.push_back(std::vector<int>{0, 0, width, 0, 0, height, width, height});
        }

        // Visualization(img, boxes);

        OCRPredictResult *results = new OCRPredictResult[boxes.size()];

        if (rec)
        {
            std::vector<std::string> rec_text;
            std::vector<float> rec_text_score;

            this->recognizer_->Run(img, boxes, this->charactor_dict,
                                   rec_text, rec_text_score);

            for (size_t i = 0; i < boxes.size(); i++)
            {
                memcpy(results[i].box, boxes[i].data(), sizeof(int) * 8);
                results[i].text = toCstr(rec_text[i]);
                results[i].score = rec_text_score[i];
            }
        }

        return {results, boxes.size()};
    }
}
