#include "opencv2/imgproc.hpp"

#include "include/ocr_det.h"
#include "include/post_process.h"
#include "include/pre_process.h"

DBDetector::DBDetector(const Flags &flags) {
    this->det_db_use_dilate = flags.det_db_use_dilate;
    this->det_db_box_thresh = flags.det_db_box_thresh;
    this->det_db_unclip_ratio = flags.det_db_unclip_ratio;
    this->det_use_polygon_score = flags.det_use_polygon_score;
    this->det_db_thresh = flags.det_db_thresh;
}

mindspore::Status DBDetector::RunPredict(const cv::Mat &img,
                                         std::vector<std::vector<int>> &boxes) {
    cv::Mat img_fp;

    std::vector<float> ratio_hw;
    DetResizeImg(img, 960, ratio_hw).convertTo(img_fp, CV_32FC3, 1.0 / 255.f);

    // Prepare input data from image
    std::vector<int64_t> resize_shape = {1, img_fp.rows, img_fp.cols, 3};
    std::vector<std::vector<int64_t>> new_shapes;
    new_shapes.push_back(resize_shape);
    auto resize_res = this->ResizeInputsTensorShape(new_shapes);
    if (resize_res != mindspore::kSuccess) {
        return resize_res;
    }

    const float *dimg = reinterpret_cast<const float *>(img_fp.data);
    auto set_res = this->SetInputData(dimg, img_fp.rows * img_fp.cols * 3);
    if (set_res != mindspore::kSuccess) {
        return set_res;
    }

    mindspore::MSTensor out_tensor;
    auto predict_res = this->Predict(out_tensor);
    if (predict_res != mindspore::kSuccess) {
        return predict_res;
    }

    auto out_shape = out_tensor.Shape();
    auto out_data = reinterpret_cast<float *>(out_tensor.MutableData());

    // Save output
    float pred[out_tensor.ElementNum()];
    unsigned char cbuf[out_tensor.ElementNum()];

    for (int64_t i = 0; i < out_tensor.ElementNum(); i++) {
        pred[i] = out_data[i];
        cbuf[i] = static_cast<unsigned char>((out_data[i]) * 255);
    }

    cv::Mat cbuf_map(out_shape[2], out_shape[3], CV_8UC1,
                     reinterpret_cast<unsigned char *>(cbuf));
    cv::Mat pred_map(out_shape[2], out_shape[3], CV_32F,
                     reinterpret_cast<float *>(pred));

    const double threshold = det_db_thresh * 255;
    const double max_value = 255;
    cv::Mat bit_map;
    cv::threshold(cbuf_map, bit_map, threshold, max_value, cv::THRESH_BINARY);
    if (det_db_use_dilate == 1) {
        cv::Mat dilation_map;
        cv::Mat dila_ele =
            cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
        cv::dilate(bit_map, dilation_map, dila_ele);
        bit_map = dilation_map;
    }

    boxes = BoxesFromBitmap(pred_map, bit_map, det_db_box_thresh,
                            det_db_unclip_ratio, det_use_polygon_score);

    boxes = FilterTagDetRes(boxes, ratio_hw[0], ratio_hw[1], img);

    return mindspore::kSuccess;
}

cv::Mat DBDetector::DetResizeImg(const cv::Mat img, int max_size_len,
                                 std::vector<float> &ratio_hw) {
    int w = img.cols;
    int h = img.rows;

    float ratio = 1.f;
    int max_wh = w >= h ? w : h;
    if (max_wh > max_size_len) {
        ratio = static_cast<float>(max_size_len) /
                static_cast<float>(h > w ? h : w);
    }

    int resize_h = static_cast<int>(float(h) * ratio);
    int resize_w = static_cast<int>(float(w) * ratio);
    if (resize_h % 32 == 0)
        resize_h = resize_h;
    else if (resize_h / 32 < 1 + 1e-5)
        resize_h = 32;
    else
        resize_h = (resize_h / 32 - 1) * 32;

    if (resize_w % 32 == 0)
        resize_w = resize_w;
    else if (resize_w / 32 < 1 + 1e-5)
        resize_w = 32;
    else
        resize_w = (resize_w / 32 - 1) * 32;

    cv::Mat resize_img;
    cv::resize(img, resize_img, cv::Size(resize_w, resize_h));

    ratio_hw.push_back(static_cast<float>(resize_h) / static_cast<float>(h));
    ratio_hw.push_back(static_cast<float>(resize_w) / static_cast<float>(w));

    return resize_img;
}

mindspore::Status DBDetector::SetInputData(const float *data,
                                           size_t data_size) {
    if (data == nullptr) {
        return mindspore::kLiteInputParamInvalid;
    }

    auto in_tensor = this->GetInputTensor(0);
    if (in_tensor == nullptr) {
        return mindspore::kLiteInputParamInvalid;
    }

    if (in_tensor.DataSize() != data_size * sizeof(float)) {
        return mindspore::kLiteInputParamInvalid;
    }

    auto input_data = in_tensor.MutableData();
    std::vector<float> mean = {0.406f, 0.485f, 0.456f};              // BGR
    std::vector<float> scale = {1 / 0.225f, 1 / 0.229f, 1 / 0.224f}; // BGR
    AvxMeanScale(data, static_cast<float *>(input_data), data_size, mean,
                 scale);
    return mindspore::kSuccess;
}