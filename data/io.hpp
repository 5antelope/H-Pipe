#include "common.hpp"
#include "data.pb.h"

using ::google::protobuf::Message;

cv::Mat ReadImageToCVMat(const string& filename, const int height, const int width, const bool is_color);

cv::Mat ReadImageToCVMat(const string& filename, const int height, const int width);

cv::Mat ReadImageToCVMat(const string& filename, const bool is_color);

cv::Mat ReadImageToCVMat(const string& filename);
