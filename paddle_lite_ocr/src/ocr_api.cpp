#include "opencv2/imgcodecs.hpp"

#include <include/paddleocr.h>

using namespace PaddleOCR;

#include <include/ocr_api.h>

#ifdef __cplusplus
extern "C"
{
#endif
    PPOCR *init_ppocr(const Flags &flags)
    {
        return new PPOCR(flags);
    }

    void release_ppocr(PPOCR *ppocr)
    {
        delete ppocr;
    }

    const char *ocr(PPOCR *ppocr, const char *img_path,
                    OCRPredictResultArray &array,
                    bool det, bool rec)
    {
        if (!ppocr)
        {
            return toCstr("[Error] null ptr ppocr");
        }

        cv::String img_name(img_path);
        cv::Mat img = cv::imread(img_name, cv::IMREAD_COLOR);
        if (!img.data)
        {
            return toCstr("[ERROR] image read failed");
        }

        array = ppocr->ocr(img, det, rec);

        return nullptr;
    }

    Flags flags_default()
    {
        Flags flags;

        flags.max_side_len = 960;
        flags.runtime_device = toCstr("x86");
        flags.precision = toCstr("FP32"); // Only support FP32 or INT8
        flags.num_threads = 4;

        // det
        flags.det_model_file = toCstr("");
        flags.det_db_use_dilate = 0;
        flags.det_db_box_thresh = 0.5;
        flags.det_db_unclip_ratio = 1.6;
        flags.det_use_polygon_score = 1;
        flags.det_db_thresh = 0.3;

        // rec
        flags.rec_model_file = toCstr("");
        flags.dict_path = toCstr("");
        flags.rec_image_height = 48;
        flags.use_direction_classify = 1;

        // cls
        flags.cls_model_file = toCstr("");

        flags.det_mf = "/home/hmk/Documents/ocr_model/ch_PP-OCRv4_det_infer/model";
        flags.det_pf = "/home/hmk/Documents/ocr_model/ch_PP-OCRv4_det_infer/params";
        flags.rec_mf = "/home/hmk/Documents/ocr_model/ch_PP-OCRv4_rec_infer/model";
        flags.rec_pf = "/home/hmk/Documents/ocr_model/ch_PP-OCRv4_rec_infer/params";
        flags.cls_mf = "/home/hmk/Documents/ocr_model/ch_ppocr_mobile_v2.0_cls_infer/model";
        flags.cls_pf = "/home/hmk/Documents/ocr_model/ch_ppocr_mobile_v2.0_cls_infer/params";
        return flags;
    }

    void release_flags(Flags &flags)
    {
        delete[] flags.runtime_device;
        delete[] flags.precision;
        delete[] flags.det_model_file;
        delete[] flags.rec_model_file;
        delete[] flags.dict_path;
        delete[] flags.cls_model_file;
    }

#ifdef __cplusplus
}
#endif