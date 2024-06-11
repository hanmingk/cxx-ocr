#include <immintrin.h>
#include <math.h>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

#include "include/ocr_cls.h"
#include "include/pre_process.h"

void AvxMeanScale(const float *din, float *dout, size_t size,
                  const std::vector<float> &mean,
                  const std::vector<float> &scale) {
    __m256 vmean0 = _mm256_set_ps(mean[0], mean[1], mean[2], mean[0], mean[1],
                                  mean[2], mean[0], mean[1]);
    __m256 vmean1 = _mm256_set_ps(mean[2], mean[0], mean[1], mean[2], mean[0],
                                  mean[1], mean[2], mean[0]);
    __m256 vmean2 = _mm256_set_ps(mean[1], mean[2], mean[0], mean[1], mean[2],
                                  mean[0], mean[1], mean[2]);

    __m256 vscale0 = _mm256_set_ps(scale[0], scale[1], scale[2], scale[0],
                                   scale[1], scale[2], scale[0], scale[1]);
    __m256 vscale1 = _mm256_set_ps(scale[2], scale[0], scale[1], scale[2],
                                   scale[0], scale[1], scale[2], scale[0]);
    __m256 vscale2 = _mm256_set_ps(scale[1], scale[2], scale[0], scale[1],
                                   scale[2], scale[0], scale[1], scale[2]);

    size /= 3;
    size_t i = 0;
    for (; i <= size - 8; i += 8) {
        __m256 vin0 = _mm256_loadu_ps(din);
        __m256 vin1 = _mm256_loadu_ps(din + 8);
        __m256 vin2 = _mm256_loadu_ps(din + 16);

        __m256 vsub0 = _mm256_sub_ps(vin0, vmean0);
        __m256 vsub1 = _mm256_sub_ps(vin1, vmean1);
        __m256 vsub2 = _mm256_sub_ps(vin2, vmean2);

        __m256 vs0 = _mm256_mul_ps(vsub0, vscale0);
        __m256 vs1 = _mm256_mul_ps(vsub1, vscale1);
        __m256 vs2 = _mm256_mul_ps(vsub2, vscale2);

        _mm256_storeu_ps(dout, vs0);
        _mm256_storeu_ps(dout + 8, vs1);
        _mm256_storeu_ps(dout + 16, vs2);

        din += 24;
        dout += 24;
    }

    for (; i < size; i++) {
        *(dout++) = (*(din++) - mean[0]) * scale[0];
        *(dout++) = (*(din++) - mean[1]) * scale[1];
        *(dout++) = (*(din++) - mean[2]) * scale[2];
    }
}

cv::Mat GetRotateCropImage(const cv::Mat &img, const std::vector<int> &box) {
    std::vector<int> points = box;

    int x_collect[4] = {box[0], box[2], box[4], box[6]};
    int y_collect[4] = {box[1], box[3], box[5], box[7]};
    int left = int(*std::min_element(x_collect, x_collect + 4));
    int right = int(*std::max_element(x_collect, x_collect + 4));
    int top = int(*std::min_element(y_collect, y_collect + 4));
    int bottom = int(*std::max_element(y_collect, y_collect + 4));

    cv::Mat img_crop;
    img(cv::Rect(left, top, right - left, bottom - top)).copyTo(img_crop);

    for (size_t i = 0; i < 4; i++) {
        points[i * 2] -= left;
        points[i * 2 + 1] -= top;
    }

    int img_crop_width = static_cast<int>(
        sqrt(pow(points[0] - points[2], 2) + pow(points[1] - points[3], 2)));
    int img_crop_height = static_cast<int>(
        sqrt(pow(points[0] - points[6], 2) + pow(points[1] - points[7], 2)));

    cv::Point2f pts_std[4];
    pts_std[0] = cv::Point2f(0., 0.);
    pts_std[1] = cv::Point2f(img_crop_width, 0.);
    pts_std[2] = cv::Point2f(img_crop_width, img_crop_height);
    pts_std[3] = cv::Point2f(0.f, img_crop_height);

    cv::Point2f pointsf[4];
    pointsf[0] = cv::Point2f(points[0], points[1]);
    pointsf[1] = cv::Point2f(points[2], points[3]);
    pointsf[2] = cv::Point2f(points[4], points[5]);
    pointsf[3] = cv::Point2f(points[6], points[7]);

    cv::Mat M = cv::getPerspectiveTransform(pointsf, pts_std);

    cv::Mat dst_img;
    cv::warpPerspective(img_crop, dst_img, M,
                        cv::Size(img_crop_width, img_crop_height),
                        cv::BORDER_REPLICATE);
    const float ratio = 1.5;
    if (static_cast<float>(dst_img.rows) >=
        static_cast<float>(dst_img.cols) * ratio) {
        cv::Mat srcCopy = cv::Mat(dst_img.rows, dst_img.cols, dst_img.depth());
        cv::transpose(dst_img, srcCopy);
        cv::flip(srcCopy, srcCopy, 0);
        return srcCopy;
    } else {
        return dst_img;
    }
}