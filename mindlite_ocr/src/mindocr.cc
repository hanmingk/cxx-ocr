#include "include/mindocr.h"
#include "include/utils.h"

MindOCR::MindOCR() {}

mindspore::Status MindOCR::Init(const Flags &flags) {
    if (flags.det) {
        this->detector_.reset(new DBDetector(flags));
        auto res = this->detector_->InitModel(flags.det_model_file);
        if (res != mindspore::kSuccess) {
            return res;
        }
    }

    if (flags.rec) {
        this->recognize_.reset(new CRNNRecognize(flags));
        auto res = this->recognize_->InitModel(flags.rec_model_file);
        if (res != mindspore::kSuccess) {
            return res;
        }
    }

    if (flags.cls) {
        this->classifier_.reset(new Classifier(flags.cls_thresh));
        auto res = this->classifier_->InitModel(flags.cls_model_file);
        if (res != mindspore::kSuccess) {
            return res;
        }
    }

    return mindspore::kSuccess;
}

mindspore::Status
MindOCR::DetPredict(const cv::Mat &img,
                    std::vector<std::vector<int32_t>> &boxes) {
    if (this->detector_ == nullptr) {
        return mindspore::kLiteNullptr;
    }

    auto res = this->detector_->RunPredict(img, boxes);
    if (res != mindspore::kSuccess) {
        return res;
    }

    return mindspore::kSuccess;
}

mindspore::Status MindOCR::RecPredict(const cv::Mat &img, std::string &rec_text,
                                      float &rec_text_score, bool cls) {
    if (this->recognize_ == nullptr) {
        return mindspore::kLiteNullptr;
    }

    cv::Mat rec_img = img;
    if (cls) {
        if (this->classifier_ == nullptr) {
            return mindspore::kLiteNullptr;
        }

        bool is_roate = false;
        auto res = this->classifier_->RunPredict(rec_img, is_roate);
        if (res != mindspore::kSuccess) {
            return res;
        }

        if (is_roate) {
            rec_img = Classifier::ClsRotateImg(rec_img);
        }
    }

    auto rec_res =
        this->recognize_->RunPredict(rec_img, rec_text, rec_text_score);
    if (rec_res != mindspore::kSuccess) {
        return rec_res;
    }

    return mindspore::kSuccess;
}
