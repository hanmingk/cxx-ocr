#include <include/ocr_wrapper.h>
#include <include/flags.h>

#include <iostream>
#include <vector>

void print_results(const OCRPredictResultMatrix &matrix) {
    for (int i = 0; i < matrix.arrays[0].len; i++)
    {
        std::cout << i << "\t";
        // det
        int *boxes = matrix.arrays[0].results[i].box;
        for (int n = 0; n < 4; n++)
        {
            std::cout << '[' << boxes[n * 2] << ',' << boxes[n * 2 + 1] << "]";
            if (n != 3)
            {
                std::cout << ',';
            }
        }
        std::cout << "] ";
        // rec
        if (matrix.arrays[0].results[0].score != -1.0)
        {
            std::cout << "rec text: " << matrix.arrays[0].results[i].text
                      << " rec score: " << matrix.arrays[0].results[i].score << " ";
        }

        // cls
        if (matrix.arrays[0].results[i].cls_label != -1)
        {
            std::cout << "cls label: " << matrix.arrays[0].results[i].cls_label
                      << " cls score: " << matrix.arrays[0].results[i].cls_score;
        }
        std::cout << std::endl;
    }
}

int main()
{
    Flags *flags = flags_default();
    delete[] flags->det_model_dir; 
    flags->det_model_dir = to_cstr("/app/ch_PP-OCRv4_det_infer");
    delete[] flags->rec_model_dir; 
    flags->rec_model_dir = to_cstr("/app/ch_PP-OCRv4_rec_infer");

    PPOCR *ppocr = init_ppocr(*flags);

    const char *img_paths[] = {"/app/PaddleOCR-2.7.5/deploy/cpp_infer1/login.jpg"};

    OCRPredictResultMatrix *matrix;
    const char *msg = paddle_ocr(ppocr, matrix, img_paths, 1, flags->det, flags->rec, flags->cls);

    print_results(*matrix);

    release_ppocr(ppocr);

    release_flags(flags);

    return 0;
}
