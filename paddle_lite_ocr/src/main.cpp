#include <include/ocr_api.h>
#include <include/utils.h>
#include <stdio.h>

#include <iostream>

void print_results(const OCRPredictResultArray &array) {
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
        if (array.results[0].score != -1.0) {
            std::cout << "rec text: " << array.results[i].text
                      << " rec score: " << array.results[i].score << " ";
        }
        std::cout << std::endl;
    }
}

int main(int, char **) {
    Flags flags = flags_default();
    delete[] flags.det_model_file;
    flags.det_model_file = toCstr(
        "/home/hmk/Documents/ocr_model/ch_PP-OCRv4_det_infer/"
        "ch_PP-OCRv4_det_opt.nb");
    delete[] flags.rec_model_file;
    flags.rec_model_file = toCstr(
        "/home/hmk/Documents/ocr_model/ch_PP-OCRv4_rec_infer/"
        "ch_PP-OCRv4_rec_opt.nb");
    delete[] flags.cls_model_file;
    flags.cls_model_file = toCstr(
        "/home/hmk/Documents/ocr_model/ch_ppocr_mobile_v2.0_cls_infer/"
        "ch_ppocr_mobile_v2.0_cls_opt.nb");
    delete[] flags.dict_path;
    flags.dict_path =
        toCstr("/home/hmk/code/cpp/paddle_lite_ocr/asset/ppocr_keys_v1.txt");

    PPOCR *ppocr = init_ppocr(flags);

    OCRPredictResultArray array;
    ocr(ppocr, "/home/hmk/code/cpp/paddle_lite_ocr/asset/11.jpg", array);
    print_results(array);

    release_ppocr(ppocr);
    release_flags(flags);
}
