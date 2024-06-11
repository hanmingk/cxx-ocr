#pragma once

char *to_cstr(const char *src);

extern "C"
{
    typedef struct
    {
        // common args
        bool use_gpu;
        bool use_tensorrt;
        int gpu_id;
        int gpu_mem;
        int cpu_threads;
        bool enable_mkldnn;
        char *precision;
        bool benchmark;
        char *output;
        char *image_dir;
        char *type;
        // detection related
        char *det_model_dir;
        char *limit_type;
        int limit_side_len;
        double det_db_thresh;
        double det_db_box_thresh;
        double det_db_unclip_ratio;
        bool use_dilation;
        char *det_db_score_mode;
        bool visualize;
        // classification related
        bool use_angle_cls;
        char *cls_model_dir;
        double cls_thresh;
        int cls_batch_num;
        // recognition related
        char *rec_model_dir;
        int rec_batch_num;
        char *rec_char_dict_path;
        int rec_img_h;
        int rec_img_w;
        // layout model related
        char *layout_model_dir;
        char *layout_dict_path;
        double layout_score_threshold;
        double layout_nms_threshold;
        // structure model related
        char *table_model_dir;
        int table_max_len;
        int table_batch_num;
        char *table_char_dict_path;
        bool merge_no_span_structure;
        // forward related
        bool det;
        bool rec;
        bool cls;
        bool table;
        bool layout;
    } Flags;

    Flags *flags_default();

    void release_flags(Flags *flags);
}