#pragma once

#include <string>

#include "api/status.h"

mindspore::Status RealPath(std::string &path);

mindspore::Status ReadFile(const std::string &file_path, char *&buf,
                           size_t *size);

char *toCstr(const std::string &str);