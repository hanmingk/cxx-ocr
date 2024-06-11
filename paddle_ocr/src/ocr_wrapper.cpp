#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <vector>

#include <include/paddleocr.h>
#include <include/paddlestructure.h>

using namespace PaddleOCR;

#include <include/ocr_wrapper.h>

static char *strdup_cpp(const std::string &str)
{
    char *cstr = new char[str.length() + 1];
    str.copy(cstr, str.length());
    cstr[str.length()] = '\0';
    return cstr;
}

static void copy_box_data(const std::vector<std::vector<int>> &box, int *box_data)
{
    for (int n = 0; n < 4; ++n)
    {
        box_data[n * 2] = box[n][0];
        box_data[n * 2 + 1] = box[n][1];
    }
}

static void ocr_result_transform(const OCRPredictResult &ocr_res, COCRPredictResult &other)
{
    copy_box_data(ocr_res.box, other.box);
    other.text = strdup_cpp(ocr_res.text);
    other.score = ocr_res.score;
    other.cls_score = ocr_res.cls_score;
    other.cls_label = ocr_res.cls_label;
}

extern "C"
{
    PPOCR *init_ppocr(const Flags &flags)
    {
        return new PaddleOCR::PPOCR(flags);
    }

    const char *paddle_ocr(
        PPOCR *ppocr, OCRPredictResultMatrix *&matrix,
        const char **img_paths, std::size_t num,
        bool det, bool rec, bool cls)
    {
        if (!ppocr)
        {
            return "[ERROR] null ptr ppocr";
        }

        ppocr->reset_timer();

        std::vector<cv::Mat> img_list;
        for (size_t i = 0; i < num; ++i)
        {
            cv::String img_name(img_paths[i]);
            cv::Mat img = cv::imread(img_name, cv::IMREAD_COLOR);
            if (!img.data)
            {
                return "[ERROR] image read failed";
            }
            img_list.push_back(img);
        }

        std::vector<std::vector<OCRPredictResult>> ocr_results =
            ppocr->ocr(img_list, det, rec, cls);

        matrix = new OCRPredictResultMatrix();

        matrix->arrays = new OCRPredictResultArray[ocr_results.size()];
        matrix->len = ocr_results.size();

        for (size_t i = 0; i < ocr_results.size(); ++i)
        {
            COCRPredictResult *res_array = new COCRPredictResult[ocr_results[i].size()];

            for (size_t j = 0; j < ocr_results[i].size(); ++j)
            {
                const auto &ocr_res = ocr_results[i][j];
                ocr_result_transform(ocr_res, res_array[j]);
            }

            matrix->arrays[i].results = res_array;
            matrix->arrays[i].len = ocr_results[i].size();
        }

        return nullptr;
    }

    void release_ppocr(PPOCR *ppocr)
    {
        if (ppocr)
        {
            delete ppocr;
        }
    }

    PaddleStructure *init_pstru(const Flags &flags)
    {
        return new PaddleOCR::PaddleStructure(flags);
    }

    const char *paddle_stru(
        PaddleStructure *pstru, StructurePredictResultMatrix *&matrix,
        const char **img_paths, std::size_t num,
        bool layout, bool table, bool ocr)
    {
        if (!pstru)
        {
            return "Null ptr pstructure";
        }

        matrix = new StructurePredictResultMatrix();

        matrix->arrays = new StructurePredictResultArray[num];
        matrix->len = num;

        for (size_t i = 0; i < num; ++i)
        {
            cv::String img_name(img_paths[i]);
            cv::Mat img = cv::imread(img_name, cv::IMREAD_COLOR);
            if (!img.data)
            {
                for (size_t x = 0; x < matrix->len; ++x)
                {
                    for (size_t y = 0; y < matrix->arrays[x].len; ++y)
                    {
                        release_stru_result(&(matrix->arrays[x].results[y]));
                    }
                    delete[] matrix->arrays[x].results;
                }
                delete[] matrix->arrays;
                matrix = nullptr;
                return "[ERROR] image read failed!";
            }

            std::vector<StructurePredictResult> structure_results =
                pstru->structure(img, layout, table, ocr);

            CStructurePredictResult *res_array =
                new CStructurePredictResult[structure_results.size()];

            for (size_t j = 0; j < structure_results.size(); ++j)
            {
                const auto &cur_res = structure_results[j];
                const float *box_ptr = cur_res.box.data();
                std::copy(box_ptr, box_ptr + 4, res_array[j].box);

                res_array[j].cell_box = new int*[cur_res.cell_box.size()];
                res_array[j].cell_len = cur_res.cell_box.size();

                if (res_array[j].cell_len > 0)
                {
                    res_array[j].box_len = cur_res.cell_box[0].size();
                }
                for (size_t n = 0; n < cur_res.cell_box.size(); ++n)
                {
                    res_array[j].cell_box[n] = new int[cur_res.cell_box[n].size()];
                    std::copy(cur_res.cell_box[n].data(),
                              cur_res.cell_box[n].data() + cur_res.cell_box[n].size(),
                              res_array[j].cell_box[n]);
                }

                res_array[j].type = strdup_cpp(cur_res.type);

                res_array[j].text_res = new COCRPredictResult[cur_res.text_res.size()];
                res_array[j].res_len = cur_res.text_res.size();
                for (size_t n = 0; n < cur_res.text_res.size(); ++n)
                {
                    ocr_result_transform(cur_res.text_res[n], res_array[j].text_res[n]);
                }

                res_array[j].html = strdup_cpp(cur_res.html);
                res_array[j].html_score = cur_res.html_score;
                res_array[j].confidence = cur_res.confidence;
            }

            matrix->arrays[i].results = res_array;
            matrix->arrays[i].len = structure_results.size();
        }

        return nullptr;
    }

    void release_pstru(PaddleStructure *pstru)
    {
        if (pstru)
        {
            delete pstru;
        }
    }

    void release_ocr_result(COCRPredictResult *ppocr)
    {
        if (ppocr)
        {
            delete[] ppocr->text;
        }
    }

    void release_stru_result(CStructurePredictResult *pstru)
    {
        for (size_t i = 0; i < pstru->cell_len; ++i)
        {
            delete[] pstru->cell_box[i];
        }
        delete[] pstru->cell_box;

        delete[] pstru->type;

        for (size_t i = 0; i < pstru->res_len; ++i)
        {
            release_ocr_result(&(pstru->text_res[i]));
        }
        delete[] pstru->text_res;

        delete[] pstru->html;
    }
}