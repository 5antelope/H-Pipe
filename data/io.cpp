#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <stdint.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "common.hpp"
#include "io.hpp"
#include "data.pb.h"

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte

cv::Mat ReadImageToCVMat(const string& filename, const int height, const int width, const bool is_color) {
    cv::Mat cv_img;

    int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
            CV_LOAD_IMAGE_GRAYSCALE);

    cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);

    if (!cv_img_origin.data) {
        LOG(ERROR) << "Could not open or find file " << filename;
        return cv_img_origin;
    }

    if (height > 0 && width > 0) {
        cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
    } else {
        cv_img = cv_img_origin;
    }

    return cv_img;
}

cv::Mat ReadImageToCVMat(const string& filename, const int height, const int width) {
    return ReadImageToCVMat(filename, height, width, true);
}

cv::Mat ReadImageToCVMat(const string& filename, const bool is_color) {
    return ReadImageToCVMat(filename, 0, 0, is_color);
}

cv::Mat ReadImageToCVMat(const string& filename) {
    return ReadImageToCVMat(filename, 0, 0, true);
}

// Do the file extension and encoding match?
static bool matchExt(const std::string & fn, std::string en) {
    size_t p = fn.rfind('.');

    std::string ext = p != fn.npos ? fn.substr(p) : fn;
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    std::transform(en.begin(), en.end(), en.begin(), ::tolower);

    if ( ext == en )
        return true;
    if ( en == "jpg" && ext == "jpeg" )
        return true;

    return false;
}
