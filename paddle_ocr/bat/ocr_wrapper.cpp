#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <vector>

#include <include/paddleocr.h>
#include <include/paddlestructure.h>

using namespace PaddleOCR;

#include <include/ocr_wrapper.h>

extern "C"
{

    PPOCR *new_ppocr(const Flags &flags)
    {
        return new PPOCR(flags);
    }

    const char *paddle_ocr(
        PPOCR *ppocr,
        ImgOCRPredictResult **results,
        const char **img_paths, std::size_t num,
        bool det, bool rec, bool cls)
    {
        if (!ppocr)
        {
            return "Null ptr ppocr";
        }

        ppocr->reset_timer();

        std::vector<cv::Mat> img_list;
        for (int i = 0; i < num; ++i)
        {
            cv::String img_name(img_paths[i]);
            cv::Mat img = cv::imread(img_name, cv::IMREAD_COLOR);
            if (!img.data)
            {
                return "[ERROR] image read failed!";
            }
            img_list.push_back(img);
        }

        std::vector<std::vector<OCRPredictResult>> ocr_results =
            ppocr->ocr(img_list, det, rec, cls);
        
        *results = new ImgOCRPredictResult[ocr_results.size()];
        
        for (int i = 0; i < ocr_results.size(); ++i)
        {
            PPOCRPredictResult *pp_res = new PPOCRPredictResult[ocr_results[i].size()];
            for (int j = 0; j < ocr_results[i].size(); ++j)
            {
                const auto &ocr_res = ocr_results[i][j];
                pp_res[j].box = new int[8];
                for (int n = 0; n < 4; ++n) {
                    pp_res[j].box[n * 2] = ocr_res.box[n][0];
                    pp_res[j].box[n * 2 + 1] = ocr_res.box[n][1];
                }
                pp_res[j].text = new char[ocr_res.text.length() + 1];
                ocr_res.text.copy(pp_res[j].text, ocr_res.text.length());
                pp_res[j].text[ocr_res.text.length()] = '\0';
                pp_res[j].score = ocr_res.score;
                pp_res[j].cls_score = ocr_res.cls_score;
                pp_res[j].cls_label = ocr_res.cls_label;
            }
            (*results)[i].results = pp_res;
            (*results)[i].len = ocr_results[i].size();
        }

        return nullptr;
    }

    void destory_ppocr(PPOCR *ppocr)
    {
        delete ppocr;
    }

    PaddleStructure *new_pstructure(const Flags &flags)
    {
        return new PaddleStructure(flags);
    }

    const char *paddle_structure(
        PaddleStructure *pstructure,
        ImgStructurePredictResult *results,
        const char **img_paths, std::size_t num,
        bool layout, bool table, bool ocr)
    {
        if (!pstructure)
        {
            return "Null ptr pstructure";
        }

        for (int i = 0; i < num; i++)
        {
            cv::String img_name(img_paths[i]);
            cv::Mat img = cv::imread(img_name, cv::IMREAD_COLOR);
            if (!img.data)
            {
                return "[ERROR] image read failed!";
            }

            std::vector<StructurePredictResult> structure_results = pstructure->structure(
                img, layout, table, ocr);

            // for (int j = 0; j < structure_results.size(); j++)
            // {
            //     std::cout << j << "\ttype: " << structure_results[j].type
            //               << ", region: [";
            //     std::cout << structure_results[j].box[0] << ","
            //               << structure_results[j].box[1] << ","
            //               << structure_results[j].box[2] << ","
            //               << structure_results[j].box[3] << "], score: ";
            //     std::cout << structure_results[j].confidence << ", res: ";

            //     if (structure_results[j].type == "table")
            //     {
            //         std::cout << structure_results[j].html << std::endl;
            //         if (structure_results[j].cell_box.size() > 0 && FLAGS_visualize)
            //         {
            //             std::string file_name = Utility::basename(cv_all_img_names[i]);

            //             Utility::VisualizeBboxes(img, structure_results[j],
            //                                      FLAGS_output + "/" + std::to_string(j) +
            //                                          "_" + file_name);
            //         }
            //     }
            //     else
            //     {
            //         std::cout << "count of ocr result is : "
            //                   << structure_results[j].text_res.size() << std::endl;
            //         if (structure_results[j].text_res.size() > 0)
            //         {
            //             std::cout << "********** print ocr result "
            //                       << "**********" << std::endl;
            //             Utility::print_result(structure_results[j].text_res);
            //             std::cout << "********** end print ocr result "
            //                       << "**********" << std::endl;
            //         }
            //     }
            // }
        }

        return nullptr;
    }

    void destory_pstructure(PaddleStructure *pstructure)
    {
        delete pstructure;
    }
}
