#include "opencv2/imgproc.hpp"

#include "include/ocr_cls.h"
#include "include/pre_process.h"

mindspore::Status Classifier::RunPredict(const cv::Mat &img, bool &is_rotate) {
    cv::Mat resize_img;
    float wh_ratio =
        static_cast<float>(img.cols) / static_cast<float>(img.rows);

    resize_img = Classifier::ClsResizeImg(img);
    resize_img.convertTo(resize_img, CV_32FC3, 1 / 255.f);

    // Prepare input data from image
    std::vector<int64_t> resize_shape = {1, resize_img.rows, resize_img.cols,
                                         3};
    std::vector<std::vector<int64_t>> new_shapes;
    new_shapes.push_back(resize_shape);
    auto resize_res = this->ResizeInputsTensorShape(new_shapes);
    if (resize_res != mindspore::kSuccess) {
        return resize_res;
    }

    const float *dimg = reinterpret_cast<const float *>(resize_img.data);
    auto set_res =
        this->SetInputData(dimg, resize_img.rows * resize_img.cols * 3);
    if (set_res != mindspore::kSuccess) {
        return set_res;
    }

    // Get output and run postprocess
    mindspore::MSTensor out_tensor;
    auto predict_res = this->Predict(out_tensor);
    if (predict_res != mindspore::kSuccess) {
        return predict_res;
    }

    auto out_shape = out_tensor.Shape();
    auto out_data = reinterpret_cast<float *>(out_tensor.MutableData());

    float score = 0;
    int label = 0;
    for (int i = 0; i < out_shape[1]; i++) {
        if (out_data[i] > score) {
            score = out_data[i];
            label = i;
        }
    }

    if (label % 2 == 1 && score > this->cls_thresh_) {
        is_rotate = true;
    } else {
        is_rotate = false;
    }

    return mindspore::kSuccess;
}

cv::Mat Classifier::ClsResizeImg(const cv::Mat &img) {
    const std::vector<int> rec_image_shape{3, 48, 192};

    int imgC, imgH, imgW;
    imgC = rec_image_shape[0];
    imgH = rec_image_shape[1];
    imgW = rec_image_shape[2];

    float ratio = static_cast<float>(img.cols) / static_cast<float>(img.rows);

    int resize_w, resize_h;
    if (ceilf(imgH * ratio) > imgW)
        resize_w = imgW;
    else
        resize_w = int(ceilf(imgH * ratio));
    cv::Mat resize_img;
    cv::resize(img, resize_img, cv::Size(resize_w, imgH), 0.f, 0.f,
               cv::INTER_LINEAR);
    if (resize_w < imgW) {
        cv::copyMakeBorder(resize_img, resize_img, 0, 0, 0, imgW - resize_w,
                           cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    }
    return resize_img;
}

mindspore::Status Classifier::SetInputData(const float *data,
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
    std::vector<float> mean = {0.5f, 0.5f, 0.5f};              // BGR
    std::vector<float> scale = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f}; // BGR
    AvxMeanScale(data, static_cast<float *>(input_data), data_size, mean,
                 scale);
    return mindspore::kSuccess;
}
