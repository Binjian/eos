// Copyright 2020 Newrizon. All rights reserved.
// Use of this source code  can be found in the LICENSE file,
// which is part of this source code package.

#include "inc/xcp_data.h"

#include <string.h>

#include <iostream>
#include <iomanip>

namespace newrizon {
namespace xcp {

using std::cout;
using std::endl;

void PrintXcpInfo(XcpInfo xcp_info) {
  cout << "channel : " << xcp_info.channel << endl;
  cout << "download_can_id : " << std::hex << xcp_info.download_can_id << endl;
  cout << "upload_can_id : " << std::hex << xcp_info.upload_can_id << endl;
}

void PrintXcpData(std::vector<XcpData>* vec_data) {
  std::vector<XcpData>::const_iterator it;
  for (it = vec_data->begin(); it != vec_data->end(); ++it) {
    std::cout << "++++++++++" << std::endl;
    std::cout << "name : " << it->name << std::endl;
    std::cout << "value_type : " << it->value_type << std::endl;
    std::cout << "dim : ";
    for (int i = 0; i < it->dim.size(); i++) printf("%d ", it->dim[i]);
    std::cout << std::endl;
    std::cout << "address : " << std::endl;
    for (int i = 0; i < 4; i++) printf("%02X ", it->address[i]);
    std::cout << std::endl;
    std::cout << "values : " << std::endl;
    for (int i = 0; i < it->data.size(); i++) printf("%02X ", it->data[i]);
    std::cout << std::endl;
    if (it->value_type == "FLOAT32_IEEE") {
      auto float_vec = GetFloat32FromBuffer(it->data);
      PrintFloat32Vector(float_vec);
    }
    std::cout << std::endl << "----------" << std::endl;
  }
}

std::vector<float> GetFloat32FromBuffer(const std::vector<uint8_t>& buffer) {
  std::vector<float> ret;
  int ret_length = buffer.size() / 4;
  for (int i = 0; i < ret_length; ++i) {
    float float_value;
    memcpy(&float_value, &buffer[i * 4], sizeof(float));
    ret.push_back(float_value);
  }
  return ret;
}

std::vector<uint8_t> StringToVectorUInt8(std::string str) {
  int length = str.length();
  int i = 0;
  std::vector<uint8_t> ret;
  for (i = 0; i < length; i = i + 2) {
    ret.push_back(std::stoul(str.substr(i, 2), nullptr, 16));
  }
  return ret;
}

std::string VectorUInt8ToString(std::vector<uint8_t> buffer) {
  std::stringstream ss;
  ss << std::hex << std::setfill('0');
  for (int i = 0; i < buffer.size(); ++i) {
    ss << std::hex << std::setw(2) << static_cast<int>(buffer[i]);
  }
  return ss.str();
}

std::vector<float> StringToVectorFloat32(std::string str) {
  const int byte_length = 4;
  int length = str.length();
  int i = 0;
  std::vector<float> ret;
  std::vector<uint8_t> receive;
  for (i = 0; i < length; i = i + 2) {
    receive.push_back(std::stoul(str.substr(i, 2), nullptr, 16));
  }
  return ret;
}

// std::vector<float> VectorUInt8ToVectorFloat32(
//     const std::vector<uint8_t>& vec_data) {
//   std::vector<float> ret;
//   int size = vec_data.size();
//   int float_vector_size = size / 4;
//   for (int i = 0; i < float_vector_size; ++i) {
//     float float_value;
//     memcpy(&float_value, &vec_data[i * 4], sizeof(float));
//     ret.push_back(float_value);
//   }
// }

void PrintFloat32Vector(const std::vector<float>& vec_data) {
  std::cout << std::fixed << std::setprecision(1);
  for (int i = 0; i < vec_data.size(); ++i) {
    std::cout << vec_data[i] << ", ";
    if (i % 17 == 16) {
      std::cout << std::endl;
    }
  }
}

bool IsSame(const std::vector<uint8_t>& lvalue,
            const std::vector<uint8_t>& rvalue) {
  int l_length = lvalue.size();
  int r_length = rvalue.size();
  if (l_length != r_length) {
    return false;
  }
  for (int i = 0; i < l_length; ++i) {
    if (lvalue[i] != rvalue[i]) {
      std::cout << "value " << i << " doesn't match" << std::endl;
      printf("lvalue = %02X\n", lvalue[i]);
      printf("rvalue = %02X\n", rvalue[i]);
      return false;
    }
  }
  return true;
}

}  // namespace xcp
}  // namespace newrizon
