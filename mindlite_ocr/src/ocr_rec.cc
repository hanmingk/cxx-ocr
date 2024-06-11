#include <fstream>

#include "opencv2/imgproc.hpp"

#include "include/ocr_rec.h"
#include "include/pre_process.h"

template <class ForwardIterator>
inline size_t Argmax(ForwardIterator first, ForwardIterator last) {
    return std::distance(first, std::max_element(first, last));
}

CRNNRecognize::CRNNRecognize(const Flags &flags) {
    this->rec_image_height = flags.rec_image_height;
    this->charactor_dict = CRNNRecognize::ReadDict(flags.dict_path);
    // blank char for ctc
    this->charactor_dict.insert(charactor_dict.begin(), "#");
    this->charactor_dict.push_back(" ");
}

mindspore::Status CRNNRecognize::RunPredict(const cv::Mat &img,
                                            std::string &rec_text,
                                            float &rec_text_score) {
    cv::Mat resize_img;
    float wh_ratio =
        static_cast<float>(img.cols) / static_cast<float>(img.rows);

    resize_img = CrnnResizeImg(img, wh_ratio, this->rec_image_height);
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

    rec_text.clear();
    rec_text_score = 0.f;
    int argmax_idx;
    int last_index = 0;
    int count = 0;
    float max_value = 0.0f;

    for (int n = 0; n < out_shape[1]; n++) {
        argmax_idx = int(Argmax(&out_data[n * out_shape[2]],
                                &out_data[(n + 1) * out_shape[2]]));
        max_value = float(*std::max_element(&out_data[n * out_shape[2]],
                                            &out_data[(n + 1) * out_shape[2]]));
        if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
            rec_text_score += max_value;
            count += 1;
            rec_text += this->charactor_dict[argmax_idx];
        }
        last_index = argmax_idx;
    }
    rec_text_score /= count;

    return mindspore::kSuccess;
}

cv::Mat CRNNRecognize::CrnnResizeImg(const cv::Mat &img, float wh_ratio,
                                     int rec_img_height) {
    const std::vector<int> rec_image_shape{3, 32, 320};
    int imgC, imgH, imgW;
    imgC = rec_image_shape[0];
    imgH = rec_img_height;
    imgW = rec_image_shape[2];

    imgW = int(imgH * wh_ratio);

    float ratio = float(img.cols) / float(img.rows);
    int resize_w, resize_h;

    if (ceilf(imgH * ratio) > imgW)
        resize_w = imgW;
    else
        resize_w = int(ceilf(imgH * ratio));
    cv::Mat resize_img;
    cv::resize(img, resize_img, cv::Size(resize_w, imgH), 0.f, 0.f,
               cv::INTER_LINEAR);
    cv::copyMakeBorder(resize_img, resize_img, 0, 0, 0,
                       int(imgW - resize_img.cols), cv::BORDER_CONSTANT,
                       {127, 127, 127});
    return resize_img;
}

std::vector<std::string> CRNNRecognize::ReadDict(const std::string &dict_path) {
    std::ifstream in(dict_path);
    std::string line;
    std::vector<std::string> m_vec;
    if (in) {
        while (getline(in, line)) {
            m_vec.push_back(line);
        }
    }

    return m_vec;
}

mindspore::Status CRNNRecognize::SetInputData(const float *data,
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
