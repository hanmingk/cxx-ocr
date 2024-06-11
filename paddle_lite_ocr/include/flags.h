#pragma once

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct
    {
        int max_side_len;
        char *runtime_device;
        char *precision;
        int num_threads;

        // det
        char *det_model_file;
        int det_db_use_dilate;
        float det_db_box_thresh;
        float det_db_unclip_ratio;
        int det_use_polygon_score;
        double det_db_thresh;

        // rec
        char *rec_model_file;
        char *dict_path;
        int rec_image_height;
        int use_direction_classify;

        // cls
        char *cls_model_file;

        const char *det_mf;
        const char *det_pf;
        const char *rec_mf;
        const char *rec_pf;
        const char *cls_mf;
        const char *cls_pf;
    } Flags;

#ifdef __cplusplus
}
#endif