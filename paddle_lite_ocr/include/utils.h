#pragma once

#include <vector>
#include <cstddef>

#include "paddle_api.h"
#include "paddle_place.h"

#include "opencv2/core.hpp"

using namespace paddle::lite_api;

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct
    {
        int box[8];
        char* text;
        float score = -1.0;
    } OCRPredictResult;

    typedef struct
    {
        OCRPredictResult *results;
        size_t len;
    } OCRPredictResultArray;

#ifdef __cplusplus
}
#endif

char *toCstr(const std::string &str);

std::shared_ptr<PaddlePredictor>
loadModel(std::string model_file, int num_threads);

std::shared_ptr<PaddlePredictor>
loadModelCXX(const std::string &model_file, const std::string &params_file, int num_threads);

// On arm platform support neon
// void NeonMeanScale(const float *din, float *dout, int size,
//                    const std::vector<float> mean,
//                    const std::vector<float> scale);

// On x86 platform support avx
void AvxMeanScale(const float *din, float *dout, int size,
                  const std::vector<float>& mean,
                  const std::vector<float>& scale);