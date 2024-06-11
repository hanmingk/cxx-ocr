#include <iostream>

#include "include/ocr_model.h"
#include "include/utils.h"

MDOCRModel::MDOCRModel() {}

mindspore::Status MDOCRModel::InitModel(const std::string &model_path) {
    auto init_res = this->InitContext();
    if (init_res != mindspore::kSuccess) {
        return init_res;
    }

    auto build_res = this->CreateAndBuildModel(model_path);
    if (build_res != mindspore::kSuccess) {
        return build_res;
    }

    return mindspore::kSuccess;
}

mindspore::Status MDOCRModel::InitContext() {
    auto context = std::make_shared<mindspore::Context>();
    if (context == nullptr) {
        return mindspore::Status(mindspore::kLiteNullptr,
                                 "[MDOCRModel] New Context failed.");
    }

    auto device_info = std::make_shared<mindspore::CPUDeviceInfo>();
    if (device_info == nullptr) {
        return mindspore::Status(mindspore::kLiteNullptr,
                                 "[MDOCRModel] New CPUDeviceInfo failed.");
    }
    auto &device_list = context->MutableDeviceInfo();
    device_list.push_back(device_info);

    this->context_ = context;

    return mindspore::kSuccess;
}

mindspore::Status
MDOCRModel::CreateAndBuildModel(const std::string &model_path) {
    // Read model file.
    std::string real_path = model_path;
    auto real_res = RealPath(real_path);
    if (real_res != mindspore::kSuccess) {
        return real_res;
    }

    size_t size = 0;
    char *model_buf = nullptr;
    auto read_res = ReadFile(real_path, model_buf, &size);
    if (read_res != mindspore::kSuccess) {
        return read_res;
    }

    auto model = new (std::nothrow) mindspore::Model();
    if (model == nullptr) {
        delete[] (model_buf);
        return mindspore::Status(mindspore::kLiteNullptr,
                                 "[MDOCRModel] New Model failed.");
    }

    auto build_res =
        model->Build(model_buf, size, mindspore::kMindIR, this->context_);
    delete[] (model_buf);
    if (build_res != mindspore::kSuccess) {
        delete model;
        return mindspore::Status(mindspore::kLiteGraphFileError,
                                 "[MDOCRModel] Build model error");
    }

    this->model_ = std::shared_ptr<mindspore::Model>(model);
    return mindspore::kSuccess;
}

mindspore::Status MDOCRModel::ResizeInputsTensorShape(
    const std::vector<std::vector<int64_t>> &shapes) {
    auto inputs = this->model_->GetInputs();
    return this->model_->Resize(inputs, shapes);
}

mindspore::MSTensor MDOCRModel::GetInputTensor(size_t index) {
    auto inputs = this->model_->GetInputs();
    if (index >= inputs.size()) {
        return mindspore::MSTensor();
    }

    return inputs[index];
}

mindspore::Status MDOCRModel::Predict(mindspore::MSTensor &out_tensor) {
    auto inputs = this->model_->GetInputs();
    auto outputs = this->model_->GetOutputs();
    auto predict_res = this->model_->Predict(inputs, &outputs);
    if (predict_res != mindspore::kSuccess) {
        return predict_res;
    }

    out_tensor = outputs.front();
    return mindspore::kSuccess;
}