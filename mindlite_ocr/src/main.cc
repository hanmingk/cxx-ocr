#include <iostream>

#include "include/mindocr.h"
#include "include/ocr_api.h"
#include "include/utils.h"

void PrintResults(const OCRPredictResultArray &array) {
    for (int i = 0; i < array.len; i++) {
        std::cout << i << "\t";
        // det
        int *boxes = array.results[i].box;
        for (int n = 0; n < 4; n++) {
            std::cout << '[' << boxes[n * 2] << ',' << boxes[n * 2 + 1] << "]";
            if (n != 3) {
                std::cout << ',';
            }
        }
        std::cout << "] ";
        // rec
        std::cout << "rec text: " << array.results[i].text
                  << " rec score: " << array.results[i].score << " ";
        std::cout << std::endl;
    }
}

int main(int, char **) {
    Flags flags = flags_default();
    flags.det = true;
    delete[] (flags.det_model_file);
    flags.det_model_file =
        toCstr("/home/hmk/Documents/ocr_model/ch_PP-OCRv4_det_infer/det_db.ms");
    flags.rec = true;
    delete[] (flags.rec_model_file);
    flags.rec_model_file = toCstr(
        "/home/hmk/Documents/ocr_model/ch_PP-OCRv4_rec_infer/rec_crnn.ms");
    flags.cls = true;
    delete[] (flags.cls_model_file);
    flags.cls_model_file =
        toCstr("/home/hmk/Documents/ocr_model/ch_ppocr_mobile_v2.0_cls_infer/"
               "cls_mv4.ms");
    delete[] (flags.dict_path);
    flags.dict_path =
        toCstr("/home/hmk/code/cpp/mindlite_ocr/ppocr_keys_v1.txt");

    MindOCR *md_ocr = nullptr;
    auto status = init_mindocr(flags, md_ocr);
    if (status.code) {
        std::cout << "[Error] code: " << status.code
                  << ", Description: " << status.msg << std::endl;
        delete[] (status.msg);
        return -1;
    }

    OCRPredictResultArray array;
    auto predict_status = ocr_predict(
        md_ocr, "/home/hmk/code/cpp/mindlite_ocr/asset/login.jpg", array, true);
    if (predict_status.code) {
        std::cout << "[Error] code: " << predict_status.code
                  << ", Description: " << predict_status.msg << std::endl;
        delete[] (predict_status.msg);
        return -1;
    }

    PrintResults(array);

    flags_release(flags);
    if (md_ocr != nullptr) {
        delete md_ocr;
    }

    if (array.len != 0) {
        delete[] (array.results);
    }

    return 0;
}