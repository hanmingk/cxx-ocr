#include <include/ocr_rec.h>
#include <include/cls_process.h>
#include <include/crnn_process.h>

using namespace paddle::lite_api;

namespace PaddleOCR
{
    CRNNRecognizer::CRNNRecognizer(const Flags &flags)
    {
        this->use_direction_classify = flags.use_direction_classify;
        this->rec_image_height = flags.rec_image_height;

        this->rec_predictor_ = loadModel(flags.rec_model_file,
                                         flags.num_threads);
        if (*flags.cls_model_file != '\0')
        {
            this->cls_predictor_ = loadModel(flags.cls_model_file,
                                             flags.num_threads);
        }

        // this->rec_predictor_ = loadModelCXX(flags.rec_mf, flags.rec_pf, flags.num_threads);
        // this->cls_predictor_ = loadModelCXX(flags.cls_mf, flags.cls_pf, flags.num_threads);
    }

    void CRNNRecognizer::Run(const cv::Mat img, const std::vector<std::vector<int>> &boxes,
                             const std::vector<std::string> &charactor_dict,
                             std::vector<std::string> &rec_text, std::vector<float> &rec_text_score)
    {
        std::vector<float> mean = {0.5f, 0.5f, 0.5f};
        std::vector<float> scale = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};

        cv::Mat srcimg;
        img.copyTo(srcimg);
        cv::Mat crop_img;
        cv::Mat resize_img;

        for (int i = boxes.size() - 1; i >= 0; i--)
        {
            crop_img = GetRotateCropImage(srcimg, boxes[i]);
            if (this->use_direction_classify >= 1)
            {
                crop_img = this->RunClsModel(crop_img);
            }
            float wh_ratio =
                static_cast<float>(crop_img.cols) / static_cast<float>(crop_img.rows);

            resize_img = CrnnResizeImg(crop_img, wh_ratio, this->rec_image_height);
            resize_img.convertTo(resize_img, CV_32FC3, 1 / 255.f);

            const float *dimg = reinterpret_cast<const float *>(resize_img.data);

            std::unique_ptr<Tensor> input_tensor0(
                std::move(this->rec_predictor_->GetInput(0)));
            input_tensor0->Resize({1, 3, resize_img.rows, resize_img.cols});
            auto *data0 = input_tensor0->mutable_data<float>();

            AvxMeanScale(dimg, data0, resize_img.rows * resize_img.cols, mean, scale);
            this->rec_predictor_->Run();

            // Get output and run postprocess
            std::unique_ptr<const Tensor> output_tensor0(
                std::move(this->rec_predictor_->GetOutput(0)));
            auto *predict_batch = output_tensor0->data<float>();
            auto predict_shape = output_tensor0->shape();

            std::string str_res;
            int argmax_idx;
            int last_index = 0;
            float score = 0.f;
            int count = 0;
            float max_value = 0.0f;

            for (int n = 0; n < predict_shape[1]; n++)
            {
                argmax_idx = int(Argmax(&predict_batch[n * predict_shape[2]],
                                        &predict_batch[(n + 1) * predict_shape[2]]));
                max_value =
                    float(*std::max_element(&predict_batch[n * predict_shape[2]],
                                            &predict_batch[(n + 1) * predict_shape[2]]));
                if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index)))
                {
                    score += max_value;
                    count += 1;
                    str_res += charactor_dict[argmax_idx];
                }
                last_index = argmax_idx;
            }
            score /= count;
            rec_text.push_back(str_res);
            rec_text_score.push_back(score);
        }
    }

    cv::Mat CRNNRecognizer::RunClsModel(const cv::Mat img, const float thresh)
    {
        std::vector<float> mean = {0.5f, 0.5f, 0.5f};
        std::vector<float> scale = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};

        cv::Mat resize_img;

        float wh_ratio =
            static_cast<float>(img.cols) / static_cast<float>(img.rows);

        resize_img = ClsResizeImg(img);
        resize_img.convertTo(resize_img, CV_32FC3, 1 / 255.f);

        const float *dimg = reinterpret_cast<const float *>(resize_img.data);

        std::unique_ptr<Tensor> input_tensor0(std::move(this->cls_predictor_->GetInput(0)));
        input_tensor0->Resize({1, 3, resize_img.rows, resize_img.cols});
        auto *data0 = input_tensor0->mutable_data<float>();

        AvxMeanScale(dimg, data0, resize_img.rows * resize_img.cols, mean, scale);
        // Run CLS predictor
        this->cls_predictor_->Run();

        // Get output and run postprocess
        std::unique_ptr<const Tensor> softmax_out(
            std::move(this->cls_predictor_->GetOutput(0)));
        auto *softmax_scores = softmax_out->mutable_data<float>();
        auto softmax_out_shape = softmax_out->shape();
        float score = 0;
        int label = 0;
        for (int i = 0; i < softmax_out_shape[1]; i++)
        {
            if (softmax_scores[i] > score)
            {
                score = softmax_scores[i];
                label = i;
            }
        }
        if (label % 2 == 1 && score > thresh)
        {
            cv::Mat rotate_img;
            cv::rotate(img, rotate_img, 1);
        }
        return img;
    }
}