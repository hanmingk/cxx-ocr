#pragma once

#include <memory>

#include "api/context.h"
#include "api/model.h"
#include "api/status.h"

class MDOCRModel {
  public:
    explicit MDOCRModel();
    ~MDOCRModel() = default;

    mindspore::Status InitModel(const std::string &model_path);
    mindspore::Status InitContext();
    mindspore::Status CreateAndBuildModel(const std::string &model_path);
    mindspore::Status
    ResizeInputsTensorShape(const std::vector<std::vector<int64_t>> &shapes);
    mindspore::MSTensor GetInputTensor(size_t index);
    mindspore::Status Predict(mindspore::MSTensor &out_tensor);

  private:
    std::shared_ptr<mindspore::Model> model_;
    std::shared_ptr<mindspore::Context> context_;
};