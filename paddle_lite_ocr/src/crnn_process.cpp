#include <include/crnn_process.h> //NOLINT

#include <algorithm>
#include <memory>
#include <string>

const std::vector<int> rec_image_shape{3, 32, 320};

cv::Mat CrnnResizeImg(cv::Mat img, float wh_ratio, int rec_image_height)
{
    int imgC, imgH, imgW;
    imgC = rec_image_shape[0];
    imgH = rec_image_height;
    imgW = rec_image_shape[2];

    imgW = int(imgH * wh_ratio);

    float ratio = float(img.cols) / float(img.rows);
    int resize_w, resize_h;

    if (ceilf(imgH * ratio) > imgW)
        resize_w = imgW;
    else
        resize_w = int(ceilf(imgH * ratio));
    cv::Mat resize_img;
    cv::resize(img, resize_img, cv::Size(resize_w, imgH), 0.f, 0.f,
               cv::INTER_LINEAR);
    cv::copyMakeBorder(resize_img, resize_img, 0, 0, 0,
                       int(imgW - resize_img.cols), cv::BORDER_CONSTANT,
                       {127, 127, 127});
    return resize_img;
}

std::vector<std::string> ReadDict(const std::string &path)
{
    std::ifstream in(path);
    std::string filename;
    std::string line;
    std::vector<std::string> m_vec;
    if (in)
    {
        while (getline(in, line))
        {
            m_vec.push_back(line);
        }
    }
    else
    {
        std::cout << "no such file" << std::endl;
    }
    return m_vec;
}

cv::Mat GetRotateCropImage(cv::Mat img, const std::vector<int> &box)
{
    std::vector<int> points = box;

    int x_collect[4] = {box[0], box[2], box[4], box[6]};
    int y_collect[4] = {box[1], box[3], box[5], box[7]};
    int left = int(*std::min_element(x_collect, x_collect + 4));
    int right = int(*std::max_element(x_collect, x_collect + 4));
    int top = int(*std::min_element(y_collect, y_collect + 4));
    int bottom = int(*std::max_element(y_collect, y_collect + 4));

    cv::Mat image;
    img.copyTo(image);
    cv::Mat img_crop;
    image(cv::Rect(left, top, right - left, bottom - top)).copyTo(img_crop);

    for (size_t i = 0; i < 4; i++)
    {
        points[i * 2] -= left;
        points[i * 2 + 1] -= top;
    }

    int img_crop_width =
        static_cast<int>(sqrt(pow(points[0] - points[2], 2) +
                              pow(points[1] - points[3], 2)));
    int img_crop_height =
        static_cast<int>(sqrt(pow(points[0] - points[6], 2) +
                              pow(points[1] - points[7], 2)));

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
        static_cast<float>(dst_img.cols) * ratio)
    {
        cv::Mat srcCopy = cv::Mat(dst_img.rows, dst_img.cols, dst_img.depth());
        cv::transpose(dst_img, srcCopy);
        cv::flip(srcCopy, srcCopy, 0);
        return srcCopy;
    }
    else
    {
        return dst_img;
    }
}
