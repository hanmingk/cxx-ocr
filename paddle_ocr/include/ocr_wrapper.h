#pragma once

#include <include/flags.h>

#include <cstddef>

extern "C"
{
    typedef struct
    {
        int box[8];       // xyxyxyxy
        const char *text; // Null-terminated string
        float score;
        float cls_score;
        int cls_label;
    } COCRPredictResult;

    typedef struct
    {
        float box[4];
        int **cell_box;
        size_t cell_len;
        size_t box_len;
        const char *type;
        COCRPredictResult *text_res;
        size_t res_len;
        const char *html;
        float html_score = -1;
        float confidence;
    } CStructurePredictResult;

    typedef struct
    {
        COCRPredictResult *results;
        size_t len;
    } OCRPredictResultArray;

    typedef struct
    {
        OCRPredictResultArray *arrays;
        size_t len;
    } OCRPredictResultMatrix;

    typedef struct
    {
        CStructurePredictResult *results;
        size_t len;
    } StructurePredictResultArray;

    typedef struct
    {
        StructurePredictResultArray *arrays;
        size_t len;
    } StructurePredictResultMatrix;

    typedef struct PPOCR PPOCR;

    PPOCR *init_ppocr(const Flags &flags);

    const char *paddle_ocr(
        PPOCR *ppocr, OCRPredictResultMatrix *&matrix,
        const char **img_paths, std::size_t num,
        bool det, bool rec, bool cls);

    void release_ppocr(PPOCR *ppocr);

    typedef struct PaddleStructure PaddleStructure;

    PaddleStructure *init_pstru(const Flags &flags);

    const char *paddle_stru(
        PaddleStructure *pstru, StructurePredictResultMatrix *&matrix,
        const char **img_paths, std::size_t num,
        bool layout, bool table, bool ocr);

    void release_pstru(PaddleStructure *pstru);

    void release_ocr_result(COCRPredictResult *ppocr);

    void release_stru_result(CStructurePredictResult *pstru);
}

