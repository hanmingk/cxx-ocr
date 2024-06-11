#include <include/flags.h>

#include <cstring>

char *to_cstr(const char *src)
{
    auto len = strlen(src);
    char *cstr = new char[len + 1];
    strcpy(cstr, src);
    return cstr;
}

extern "C"
{

    Flags *flags_default()
    {
        Flags *flags = new Flags();

        flags->use_gpu = false;               // Infering with GPU or CPU.
        flags->use_tensorrt = false;          // Whether use tensorrt.
        flags->gpu_id = 0;                    // Device id of GPU to execute.
        flags->gpu_mem = 4000;                // GPU id when infering with GPU.
        flags->cpu_threads = 10;              // Num of threads with CPU.
        flags->enable_mkldnn = false;         // Whether use mkldnn with CPU.
        flags->precision = to_cstr("fp32");   // Precision be one of fp32/fp16/int8
        flags->benchmark = false;             // Whether use benchmark.
        flags->output = to_cstr("./output/"); // Save benchmark log path.
        flags->image_dir = to_cstr("");       // Dir of input image.
        flags->type =
            to_cstr("ocr"); // Perform ocr or structure, the value is selected in ['ocr','structure'].
        // detection related
        flags->det_model_dir = to_cstr("");         // Path of det inference model.
        flags->limit_type = to_cstr("max");         // limit_type of input image.
        flags->limit_side_len = 960;                // limit_side_len of input image.
        flags->det_db_thresh = 0.3;                 // Threshold of det_db_thresh.
        flags->det_db_box_thresh = 0.6;             // Threshold of det_db_box_thresh.
        flags->det_db_unclip_ratio = 1.5;           // Threshold of det_db_unclip_ratio.
        flags->use_dilation = false;                // Whether use the dilation on output map.
        flags->det_db_score_mode = to_cstr("slow"); // Whether use polygon score.
        flags->visualize = true;                    // Whether show the detection results.
        // classification related
        flags->use_angle_cls = false;       // Whether use use_angle_cls.
        flags->cls_model_dir = to_cstr(""); // Path of cls inference model.
        flags->cls_thresh = 0.9;            // Threshold of cls_thresh.
        flags->cls_batch_num = 1;           // cls_batch_num.
        // recognition related
        flags->rec_model_dir = to_cstr(""); // Path of rec inference model.
        flags->rec_batch_num = 6;           // rec_batch_num.
        flags->rec_char_dict_path =
            to_cstr("../../ppocr/utils/ppocr_keys_v1.txt"); // Path of dictionary.
        flags->rec_img_h = 48;                              // rec image height
        flags->rec_img_w = 320;                             // rec image width

        // layout model related
        flags->layout_model_dir = to_cstr(""); // Path of table layout inference model.
        flags->layout_dict_path =
            to_cstr("../../ppocr/utils/dict/layout_dict/layout_publaynet_dict.txt"); // Path of dictionary.
        flags->layout_score_threshold = 0.5;                                         // Threshold of score.
        flags->layout_nms_threshold = 0.5;                                           // Threshold of nms.
        // structure model related
        flags->table_model_dir = to_cstr("");  // Path of table struture inference model.
        flags->table_max_len = 488;            // max len size of input image.
        flags->table_batch_num = 1;            // table_batch_num.
        flags->merge_no_span_structure = true; // Whether merge <td> and </td> to <td></td>
        flags->table_char_dict_path =
            to_cstr("../../ppocr/utils/dict/table_structure_dict_ch.txt"); // Path of dictionary.

        // ocr forward related
        flags->det = true;     // Whether use det in forward.
        flags->rec = true;     // Whether use rec in forward.
        flags->cls = false;    // Whether use cls in forward.
        flags->table = false;  // Whether use table structure in forward.
        flags->layout = false; // Whether use layout analysis in forward.

        return flags;
    }

    void release_flags(Flags *flags)
    {

        delete[] flags->precision;
        delete[] flags->output;
        delete[] flags->image_dir;
        delete[] flags->type;
        delete[] flags->det_model_dir;
        delete[] flags->limit_type;
        delete[] flags->det_db_score_mode;
        delete[] flags->cls_model_dir;
        delete[] flags->rec_model_dir;
        delete[] flags->rec_char_dict_path;
        delete[] flags->layout_model_dir;
        delete[] flags->table_model_dir;
        delete[] flags->table_char_dict_path;
    }
}
