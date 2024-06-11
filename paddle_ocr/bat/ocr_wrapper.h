#pragma once

#include <include/flags.h>

#include <cstddef>

extern "C"
{
    struct PPOCRPredictResult
    {
        int *box;
        char *text;
        float score = -1.0;
        float cls_score;
        int cls_label = -1;
    };

    struct PPStructurePredictResult
    {
        float *box;
        int *cell_box;
        char *type;
        PPOCRPredictResult *text_res;
        std::size_t len;
        char *html;
        float html_score = -1;
        float confidence;
    };

    struct ImgOCRPredictResult
    {
        PPOCRPredictResult *results;
        std::size_t len;
    };

    struct ImgStructurePredictResult
    {
        PPStructurePredictResult *results;
        std::size_t len;
    };

    typedef struct PPOCR PPOCR;

    PPOCR *new_ppocr(const Flags &flags);

    const char *paddle_ocr(PPOCR *ppocr, ImgOCRPredictResult **results, const char **img_paths, std::size_t num, bool det, bool rec, bool cls);

    void destory_ppocr(PPOCR *ppocr);

    typedef struct PaddleStructure PaddleStructure;

    PaddleStructure *new_pstructure(const Flags &flags);

    const char *paddle_structure(PaddleStructure *pstructure, ImgStructurePredictResult *results, const char **img_paths, std::size_t num, bool layout, bool table, bool ocr);

    void destory_pstructure(PaddleStructure *pstructure);
}
