// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <include/paddleocr.h>

namespace PaddleOCR {

PPOCR::PPOCR(const Flags &flags) {
  std::string det_model_dir(flags.det_model_dir);
  std::string limit_type(flags.limit_type);
  std::string det_db_score_mode(flags.det_db_score_mode);
  std::string precision(flags.precision);
  std::string cls_model_dir(flags.cls_model_dir);
  std::string rec_model_dir(flags.rec_model_dir);
  std::string rec_char_dict_path(flags.rec_char_dict_path);

  if (flags.det) {
    this->detector_.reset(new DBDetector(
        det_model_dir, flags.use_gpu, flags.gpu_id, flags.gpu_mem,
        flags.cpu_threads, flags.enable_mkldnn, limit_type,
        flags.limit_side_len, flags.det_db_thresh, flags.det_db_box_thresh,
        flags.det_db_unclip_ratio, det_db_score_mode, flags.use_dilation,
        flags.use_tensorrt, precision));
  }

  if (flags.cls && flags.use_angle_cls) {
    this->classifier_.reset(new Classifier(
        cls_model_dir, flags.use_gpu, flags.gpu_id, flags.gpu_mem,
        flags.cpu_threads, flags.enable_mkldnn, flags.cls_thresh,
        flags.use_tensorrt, precision, flags.cls_batch_num));
  }
  if (flags.rec) {
    this->recognizer_.reset(new CRNNRecognizer(
        rec_model_dir, flags.use_gpu, flags.gpu_id, flags.gpu_mem,
        flags.cpu_threads, flags.enable_mkldnn, rec_char_dict_path,
        flags.use_tensorrt, precision, flags.rec_batch_num,
        flags.rec_img_h, flags.rec_img_w));
  }
}

std::vector<std::vector<OCRPredictResult>>
PPOCR::ocr(std::vector<cv::Mat> img_list, bool det, bool rec, bool cls) {
  std::vector<std::vector<OCRPredictResult>> ocr_results;

  if (!det) {
    std::vector<OCRPredictResult> ocr_result;
    ocr_result.resize(img_list.size());
    if (cls && this->classifier_) {
      this->cls(img_list, ocr_result);
      for (int i = 0; i < img_list.size(); i++) {
        if (ocr_result[i].cls_label % 2 == 1 &&
            ocr_result[i].cls_score > this->classifier_->cls_thresh) {
          cv::rotate(img_list[i], img_list[i], 1);
        }
      }
    }
    if (rec) {
      this->rec(img_list, ocr_result);
    }
    for (int i = 0; i < ocr_result.size(); ++i) {
      std::vector<OCRPredictResult> ocr_result_tmp;
      ocr_result_tmp.push_back(ocr_result[i]);
      ocr_results.push_back(ocr_result_tmp);
    }
  } else {
    for (int i = 0; i < img_list.size(); ++i) {
      std::vector<OCRPredictResult> ocr_result =
          this->ocr(img_list[i], true, rec, cls);
      ocr_results.push_back(ocr_result);
    }
  }
  return ocr_results;
}

std::vector<OCRPredictResult> PPOCR::ocr(cv::Mat img, bool det, bool rec,
                                         bool cls) {

  std::vector<OCRPredictResult> ocr_result;
  // det
  this->det(img, ocr_result);
  // crop image
  std::vector<cv::Mat> img_list;
  for (int j = 0; j < ocr_result.size(); j++) {
    cv::Mat crop_img;
    crop_img = Utility::GetRotateCropImage(img, ocr_result[j].box);
    img_list.push_back(crop_img);
  }
  // cls
  if (cls && this->classifier_) {
    this->cls(img_list, ocr_result);
    for (int i = 0; i < img_list.size(); i++) {
      if (ocr_result[i].cls_label % 2 == 1 &&
          ocr_result[i].cls_score > this->classifier_->cls_thresh) {
        cv::rotate(img_list[i], img_list[i], 1);
      }
    }
  }
  // rec
  if (rec) {
    this->rec(img_list, ocr_result);
  }
  return ocr_result;
}

void PPOCR::det(cv::Mat img, std::vector<OCRPredictResult> &ocr_results) {
  std::vector<std::vector<std::vector<int>>> boxes;
  std::vector<double> det_times;

  this->detector_->Run(img, boxes, det_times);

  for (int i = 0; i < boxes.size(); i++) {
    OCRPredictResult res;
    res.box = boxes[i];
    ocr_results.push_back(res);
  }
  // sort boex from top to bottom, from left to right
  Utility::sorted_boxes(ocr_results);
  this->time_info_det[0] += det_times[0];
  this->time_info_det[1] += det_times[1];
  this->time_info_det[2] += det_times[2];
}

void PPOCR::rec(std::vector<cv::Mat> img_list,
                std::vector<OCRPredictResult> &ocr_results) {
  std::vector<std::string> rec_texts(img_list.size(), "");
  std::vector<float> rec_text_scores(img_list.size(), 0);
  std::vector<double> rec_times;
  this->recognizer_->Run(img_list, rec_texts, rec_text_scores, rec_times);
  // output rec results
  for (int i = 0; i < rec_texts.size(); i++) {
    ocr_results[i].text = rec_texts[i];
    ocr_results[i].score = rec_text_scores[i];
  }
  this->time_info_rec[0] += rec_times[0];
  this->time_info_rec[1] += rec_times[1];
  this->time_info_rec[2] += rec_times[2];
}

void PPOCR::cls(std::vector<cv::Mat> img_list,
                std::vector<OCRPredictResult> &ocr_results) {
  std::vector<int> cls_labels(img_list.size(), 0);
  std::vector<float> cls_scores(img_list.size(), 0);
  std::vector<double> cls_times;
  this->classifier_->Run(img_list, cls_labels, cls_scores, cls_times);
  // output cls results
  for (int i = 0; i < cls_labels.size(); i++) {
    ocr_results[i].cls_label = cls_labels[i];
    ocr_results[i].cls_score = cls_scores[i];
  }
  this->time_info_cls[0] += cls_times[0];
  this->time_info_cls[1] += cls_times[1];
  this->time_info_cls[2] += cls_times[2];
}

void PPOCR::reset_timer() {
  this->time_info_det = {0, 0, 0};
  this->time_info_rec = {0, 0, 0};
  this->time_info_cls = {0, 0, 0};
}

} // namespace PaddleOCR
