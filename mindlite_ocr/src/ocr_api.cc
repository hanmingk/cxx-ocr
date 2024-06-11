#include <iostream>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "include/mindocr.h"
#include "include/ocr_api.h"
#include "include/pre_process.h"
#include "include/utils.h"

inline MDOCRStatus CheckAndReadImg(MindOCR *md_ocr, const char *img_path,
                                   cv::Mat &img) {
    MDOCRStatus status;
    if (!md_ocr) {
        status.code = 2; // NullptrError
        status.msg = toCstr("Nullptr of MindOCR");
        return status;
    }

    cv::String img_name(img_path);
    img = cv::imread(img_name, cv::IMREAD_COLOR);
    if (!img.data) {
        status.code = 3; // FileReadError
        status.msg = toCstr("image read failed");
        return status;
    }

    return status;
}

void Visualization(const cv::Mat &srcimg, std::vector<std::vector<int>> boxes,
                   const char *save_path) {
    cv::Point rook_points[boxes.size()][4];
    for (int n = 0; n < boxes.size(); n++) {
        for (int m = 0; m < 4; m++) {
            rook_points[n][m] =
                cv::Point(static_cast<int>(boxes[n][m * 2]),
                          static_cast<int>(boxes[n][m * 2 + 1]));
        }
    }
    cv::Mat img_vis;
    srcimg.copyTo(img_vis);
    for (int n = 0; n < boxes.size(); n++) {
        const cv::Point *ppt[1] = {rook_points[n]};
        int npt[] = {4};
        cv::polylines(img_vis, ppt, npt, 1, 1, CV_RGB(0, 255, 0), 2, 8, 0);
    }

    cv::imwrite(save_path, img_vis);
    std::cout << "The detection visualized image saved in " << save_path
              << std::endl;
}

extern "C" {
Flags flags_default() {
    Flags flags;

    // det
    flags.det = false;
    flags.det_model_file = toCstr("");
    flags.max_size_len = 960;
    flags.det_db_use_dilate = 0;
    flags.det_db_box_thresh = 0.5;
    flags.det_db_unclip_ratio = 1.6;
    flags.det_use_polygon_score = 1;
    flags.det_db_thresh = 0.3;

    // rec
    flags.rec = false;
    flags.rec_model_file = toCstr("");
    flags.dict_path = toCstr("");
    flags.rec_image_height = 48;
    flags.use_direction_classify = 1;

    // cls
    flags.cls = false;
    flags.cls_model_file = toCstr("");
    flags.cls_thresh = 0.9;

    return flags;
}

void flags_release(Flags &flags) {
    delete[] flags.det_model_file;
    delete[] flags.rec_model_file;
    delete[] flags.dict_path;
    delete[] flags.cls_model_file;
}

MDOCRStatus init_mindocr(const Flags &flags, MindOCR *&md_ocr) {
    MDOCRStatus status;
    md_ocr = new MindOCR();

    auto res = md_ocr->Init(flags);
    if (res != mindspore::kSuccess) {
        status.code = 1; // ModelInitError
        status.msg = toCstr(res.GetErrDescription());
    }

    return status;
}

MDOCRStatus det_predict(MindOCR *md_ocr, const char *img_path,
                        DetPredictResultArray &array) {
    cv::Mat img;
    auto status = CheckAndReadImg(md_ocr, img_path, img);
    if (status.code) {
        return status;
    }

    std::vector<std::vector<int32_t>> boxes;
    auto predict_res = md_ocr->DetPredict(img, boxes);
    if (predict_res != mindspore::kSuccess) {
        status.code = 4; // ModelPredictError
        status.msg = toCstr(predict_res.GetErrDescription());
        return status;
    }

    array.boxes = new DetPredictResult[boxes.size()];
    for (size_t i = 0; i < boxes.size(); i++) {
        memcpy(array.boxes[i].box, boxes[i].data(), sizeof(int) * 8);
    }
    array.len = boxes.size();

    return status;
}

MDOCRStatus rec_predict(MindOCR *md_ocr, const char *img_path,
                        RecPredictResult &result, bool cls) {
    cv::Mat img;
    auto status = CheckAndReadImg(md_ocr, img_path, img);
    if (status.code) {
        return status;
    }

    std::string rec_text;
    float rec_text_score;
    auto predict_res = md_ocr->RecPredict(img, rec_text, rec_text_score, cls);
    if (predict_res != mindspore::kSuccess) {
        status.code = 4; // ModelPredictError
        status.msg = toCstr(predict_res.GetErrDescription());
        return status;
    }

    result.text = toCstr(rec_text);
    result.score = rec_text_score;
    return status;
}

MDOCRStatus ocr_predict(MindOCR *md_ocr, const char *img_path,
                        OCRPredictResultArray &array, bool cls) {
    cv::Mat img;
    auto status = CheckAndReadImg(md_ocr, img_path, img);
    if (status.code) {
        return status;
    }

    std::vector<std::vector<int32_t>> boxes;
    auto det_predict_res = md_ocr->DetPredict(img, boxes);
    if (det_predict_res != mindspore::kSuccess) {
        status.code = 4; // ModelPredictError
        status.msg = toCstr(det_predict_res.GetErrDescription());
        return status;
    }

    Visualization(img, boxes, "./vis.jpg");

    std::vector<std::string> rec_texts;
    std::vector<float> rec_text_scores;
    for (size_t i = 0; i < boxes.size(); i++) {
        auto crop_img = GetRotateCropImage(img, boxes[i]);

        std::string rec_text;
        float rec_text_score;
        auto rec_predict_res =
            md_ocr->RecPredict(crop_img, rec_text, rec_text_score, cls);
        if (rec_predict_res != mindspore::kSuccess) {
            status.code = 4; // ModelPredictError
            status.msg = toCstr(rec_predict_res.GetErrDescription());
            return status;
        }

        rec_texts.push_back(rec_text);
        rec_text_scores.push_back(rec_text_score);
    }

    array.results = new OCRPredictResult[boxes.size()];
    for (size_t i = 0; i < boxes.size(); i++) {
        memcpy(array.results[i].box, boxes[i].data(), sizeof(int) * 8);
        array.results[i].text = toCstr(rec_texts[i]);
        array.results[i].score = rec_text_scores[i];
    }
    array.len = boxes.size();

    return status;
}
}