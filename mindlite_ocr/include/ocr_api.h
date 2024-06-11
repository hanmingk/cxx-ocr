#pragma once

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    // det
    bool det;
    int max_size_len;
    char *det_model_file;
    int det_db_use_dilate;
    float det_db_box_thresh;
    float det_db_unclip_ratio;
    int det_use_polygon_score;
    double det_db_thresh;

    // rec
    bool rec;
    char *rec_model_file;
    char *dict_path;
    int rec_image_height;
    bool use_direction_classify;

    // cls
    bool cls;
    char *cls_model_file;
    double cls_thresh;
} Flags;

typedef struct {
    int box[8];
} DetPredictResult;

typedef struct {
    DetPredictResult *boxes;
    size_t len = 0;
} DetPredictResultArray;

typedef struct {
    char *text;
    float score = -1.0;
} RecPredictResult;

typedef struct {
    int box[8];
    char *text;
    float score = -1.0;
} OCRPredictResult;

typedef struct {
    OCRPredictResult *results;
    size_t len = 0;
} OCRPredictResultArray;

typedef struct {
    int code = 0;
    char *msg;
} MDOCRStatus;

typedef struct MindOCR MindOCR;

Flags flags_default();

void flags_release(Flags &flags);

MDOCRStatus init_mindocr(const Flags &flags, MindOCR *&md_ocr);

MDOCRStatus det_predict(MindOCR *md_ocr, const char *img_path,
                        DetPredictResultArray &array);

MDOCRStatus rec_predict(MindOCR *md_ocr, const char *img_path,
                        RecPredictResult &result, bool cls);

MDOCRStatus ocr_predict(MindOCR *md_ocr, const char *img_path,
                        OCRPredictResultArray &array, bool cls);

#ifdef __cplusplus
}
#endif