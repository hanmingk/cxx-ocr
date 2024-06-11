#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>

#include "include/utils.h"

mindspore::Status RealPath(std::string &path) {
    const size_t max = 4096;
    if (path.empty()) {
        return mindspore::Status(mindspore::kLiteError,
                                 "[RealPath] Path is empty");
    }

    if (path.length() >= max) {
        return mindspore::Status(mindspore::kLiteError,
                                 "[RealPath] Path is too long");
    }

    auto resolved_path = std::make_unique<char[]>(max);
    if (resolved_path == nullptr) {
        return mindspore::Status(mindspore::kLiteError,
                                 "[RealPath] New resolved_path failed");
    }

    char *real_path = realpath(path.c_str(), resolved_path.get());
    if (real_path == nullptr || strlen(real_path) == 0) {
        std::cerr << "file path is not valid : " << path << std::endl;
        return mindspore::Status(mindspore::kLiteError,
                                 "[RealPath] File path is not valid: " + path);
    }

    path = resolved_path.get();

    return mindspore::kSuccess;
}

mindspore::Status ReadFile(const std::string &file_path, char *&buf,
                           size_t *size) {
    if (file_path.empty()) {
        return mindspore::Status(mindspore::kLiteError,
                                 "[ReadFile] File path is empty");
    }

    std::ifstream ifs(file_path.c_str(),
                      std::ifstream::in | std::ifstream::binary);
    if (!ifs.good()) {
        return mindspore::Status(mindspore::kLiteFileError,
                                 "[ReadFile] File: " + file_path +
                                     "is not exist");
    }

    if (!ifs.is_open()) {
        return mindspore::Status(mindspore::kLiteFileError,
                                 "[ReadFile] File: " + file_path +
                                     " open failed");
    }

    ifs.seekg(0, std::ios::end);
    *size = ifs.tellg();
    std::unique_ptr<char[]> buffer(new (std::nothrow) char[*size]);
    if (buffer == nullptr) {
        ifs.close();
        return mindspore::Status(mindspore::kLiteFileError,
                                 "[ReadFile] Malloc buffer failed, file: " +
                                     file_path);
    }

    ifs.seekg(0, std::ios::beg);
    ifs.read(buffer.get(), *size);
    ifs.close();

    buf = buffer.release();
    return mindspore::kSuccess;
}

char *toCstr(const std::string &str) {
    char *cstr = new char[str.length() + 1];
    str.copy(cstr, str.length());
    cstr[str.length()] = '\0';
    return cstr;
}