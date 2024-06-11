#include <include/utils.h>
#include <iostream>

#include <immintrin.h>

std::shared_ptr<PaddlePredictor>
loadModel(std::string model_file, int num_threads)
{
    MobileConfig config;
    config.set_model_from_file(model_file);
    // config.set_model_file(model_file);
    // config.set_param_file(param_file);
    // config.set_valid_places({Place{TARGET(kX86), PRECISION(kFloat)}});

    std::shared_ptr<PaddlePredictor> predictor =
        CreatePaddlePredictor<MobileConfig>(config);

    return predictor;
}

std::shared_ptr<PaddlePredictor>
loadModelCXX(const std::string &model_file, const std::string &params_file, int num_threads)
{
    CxxConfig config;
    config.set_model_file(model_file);
    config.set_param_file(params_file);
    config.set_valid_places({Place{TARGET(kX86), PRECISION(kFloat)}});

    std::shared_ptr<PaddlePredictor> predictor =
        CreatePaddlePredictor<CxxConfig>(config);

    return predictor;
}

char *toCstr(const std::string &str)
{
    char *cstr = new char[str.length() + 1];
    str.copy(cstr, str.length());
    cstr[str.length()] = '\0';
    return cstr;
}

// fill tensor with mean and scale and trans layout: nhwc -> nchw, neon speed up
// void NeonMeanScale(const float *din, float *dout, int size,
//                    const std::vector<float> mean,
//                    const std::vector<float> scale)
// {
//     if (mean.size() != 3 || scale.size() != 3)
//     {
//         std::cerr << "[ERROR] mean or scale size must equal to 3" << std::endl;
//         exit(1);
//     }
//     float32x4_t vmean0 = vdupq_n_f32(mean[0]);
//     float32x4_t vmean1 = vdupq_n_f32(mean[1]);
//     float32x4_t vmean2 = vdupq_n_f32(mean[2]);
//     float32x4_t vscale0 = vdupq_n_f32(scale[0]);
//     float32x4_t vscale1 = vdupq_n_f32(scale[1]);
//     float32x4_t vscale2 = vdupq_n_f32(scale[2]);

//     float *dout_c0 = dout;
//     float *dout_c1 = dout + size;
//     float *dout_c2 = dout + size * 2;

//     int i = 0;
//     for (; i < size - 3; i += 4)
//     {
//         float32x4x3_t vin3 = vld3q_f32(din);
//         float32x4_t vsub0 = vsubq_f32(vin3.val[0], vmean0);
//         float32x4_t vsub1 = vsubq_f32(vin3.val[1], vmean1);
//         float32x4_t vsub2 = vsubq_f32(vin3.val[2], vmean2);
//         float32x4_t vs0 = vmulq_f32(vsub0, vscale0);
//         float32x4_t vs1 = vmulq_f32(vsub1, vscale1);
//         float32x4_t vs2 = vmulq_f32(vsub2, vscale2);
//         vst1q_f32(dout_c0, vs0);
//         vst1q_f32(dout_c1, vs1);
//         vst1q_f32(dout_c2, vs2);

//         din += 12;
//         dout_c0 += 4;
//         dout_c1 += 4;
//         dout_c2 += 4;
//     }
//     for (; i < size; i++)
//     {
//         *(dout_c0++) = (*(din++) - mean[0]) * scale[0];
//         *(dout_c1++) = (*(din++) - mean[1]) * scale[1];
//         *(dout_c2++) = (*(din++) - mean[2]) * scale[2];
//     }
// }

void AvxMeanScale(const float *din, float *dout, int size,
                  const std::vector<float> &mean,
                  const std::vector<float> &scale)
{
    const int channels = 3;
    if (mean.size() != channels || scale.size() != channels)
    {
        std::cerr << "[ERROR] mean or scale size must equal to " << channels << std::endl;
        exit(1);
    }

    __m256 vmean0 = _mm256_set1_ps(mean[0]);
    __m256 vmean1 = _mm256_set1_ps(mean[1]);
    __m256 vmean2 = _mm256_set1_ps(mean[2]);
    __m256 vscale0 = _mm256_set1_ps(scale[0]);
    __m256 vscale1 = _mm256_set1_ps(scale[1]);
    __m256 vscale2 = _mm256_set1_ps(scale[2]);

    float *dout_c0 = dout;
    float *dout_c1 = dout + size;
    float *dout_c2 = dout + size * 2;

    int i = 0;
    for (; i <= size - 8; i += 8)
    {
        __m256 vin0 = _mm256_loadu_ps(din);
        __m256 vin1 = _mm256_loadu_ps(din + 8);
        __m256 vin2 = _mm256_loadu_ps(din + 16);

        __m256 vsub0 = _mm256_sub_ps(vin0, vmean0);
        __m256 vsub1 = _mm256_sub_ps(vin1, vmean1);
        __m256 vsub2 = _mm256_sub_ps(vin2, vmean2);

        __m256 vs0 = _mm256_mul_ps(vsub0, vscale0);
        __m256 vs1 = _mm256_mul_ps(vsub1, vscale1);
        __m256 vs2 = _mm256_mul_ps(vsub2, vscale2);

        _mm256_storeu_ps(dout_c0, vs0);
        _mm256_storeu_ps(dout_c1, vs1);
        _mm256_storeu_ps(dout_c2, vs2);

        din += 24;
        dout_c0 += 8;
        dout_c1 += 8;
        dout_c2 += 8;
    }

    for (; i < size; i++)
    {
        *(dout_c0++) = (*(din++) - mean[0]) * scale[0];
        *(dout_c1++) = (*(din++) - mean[1]) * scale[1];
        *(dout_c2++) = (*(din++) - mean[2]) * scale[2];
    }
}