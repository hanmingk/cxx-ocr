#include <include/ocr_det.h>
#include <include/post_process.h>

using namespace paddle::lite_api;

namespace PaddleOCR {
DBDetector::DBDetector(const Flags &flags) {
    this->max_side_len = flags.max_side_len;
    this->det_db_thresh = flags.det_db_thresh;
    this->det_db_use_dilate = flags.det_db_use_dilate;
    this->det_db_box_thresh = flags.det_db_box_thresh;
    this->det_db_unclip_ratio = flags.det_db_unclip_ratio;
    this->det_use_polygon_score = flags.det_use_polygon_score;

    this->predictor_ = loadModel(flags.det_model_file, flags.num_threads);
    // this->predictor_ = loadModelCXX(flags.det_mf, flags.det_pf,
    // flags.num_threads);
}

void DBDetector::Run(const cv::Mat img, std::vector<std::vector<int>> &boxes) {
    cv::Mat src_img;
    img.copyTo(src_img);
    cv::Mat img_fp;

    std::vector<float> ratio_hw;
    DetResizeImg(img, this->max_side_len, ratio_hw)
        .convertTo(img_fp, CV_32FC3, 1.0 / 255.f);

    // Prepare input data from image
    std::unique_ptr<Tensor> input_tensor0(
        std::move(this->predictor_->GetInput(0)));
    input_tensor0->Resize({1, 3, img_fp.rows, img_fp.cols});
    auto *data0 = input_tensor0->mutable_data<float>();

    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> scale = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f};
    const float *dimg = reinterpret_cast<const float *>(img_fp.data);
    AvxMeanScale(dimg, data0, img_fp.rows * img_fp.cols, mean, scale);

    // Run predictor
    this->predictor_->Run();

    // Get output and post process
    std::unique_ptr<const Tensor> output_tensor(
        std::move(this->predictor_->GetOutput(0)));
    auto *outptr = output_tensor->data<float>();
    auto shape_out = output_tensor->shape();

    std::cout << "Tensor Shape: [";
    for (size_t i = 0; i < shape_out.size(); i++) {
        if (i != shape_out.size() - 1) {
            std::cout << shape_out[i] << ", ";
        } else {
            std::cout << shape_out[i] << "]" << std::endl;
        }
    }

    // Save output
    float pred[shape_out[2] * shape_out[3]];
    unsigned char cbuf[shape_out[2] * shape_out[3]];

    for (int i = 0; i < int(shape_out[2] * shape_out[3]); i++) {
        pred[i] = static_cast<float>(outptr[i]);
        cbuf[i] = static_cast<unsigned char>((outptr[i]) * 255);
    }

    cv::Mat cbuf_map(shape_out[2], shape_out[3], CV_8UC1,
                     reinterpret_cast<unsigned char *>(cbuf));
    cv::Mat pred_map(shape_out[2], shape_out[3], CV_32F,
                     reinterpret_cast<float *>(pred));

    const double threshold = this->det_db_thresh * 255;
    const double max_value = 255;
    cv::Mat bit_map;
    cv::threshold(cbuf_map, bit_map, threshold, max_value, cv::THRESH_BINARY);
    if (this->det_db_use_dilate == 1) {
        cv::Mat dilation_map;
        cv::Mat dila_ele =
            cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
        cv::dilate(bit_map, dilation_map, dila_ele);
        bit_map = dilation_map;
    }

    boxes =
        BoxesFromBitmap(pred_map, bit_map, this->det_db_box_thresh,
                        this->det_db_unclip_ratio, this->det_use_polygon_score);

    boxes = FilterTagDetRes(boxes, ratio_hw[0], ratio_hw[1], src_img);
}

// resize image to a size multiple of 32 which is required by the network
cv::Mat DetResizeImg(const cv::Mat img, int max_size_len,
                     std::vector<float> &ratio_hw) {
    int w = img.cols;
    int h = img.rows;

    float ratio = 1.f;
    int max_wh = w >= h ? w : h;
    if (max_wh > max_size_len) {
        if (h > w) {
            ratio = static_cast<float>(max_size_len) / static_cast<float>(h);
        } else {
            ratio = static_cast<float>(max_size_len) / static_cast<float>(w);
        }
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
}  // namespace PaddleOCR