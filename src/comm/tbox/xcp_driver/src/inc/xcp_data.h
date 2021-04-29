// Copyright 2020 Newrizon. All rights reserved.
// Use of this source code  can be found in the LICENSE file,
// which is part of this source code package.

#ifndef SRC_INC_XCP_DATA_H_
#define SRC_INC_XCP_DATA_H_

#include <vector>
#include <cstdint>
#include <string>

namespace newrizon {
namespace xcp {

struct XcpInfo{
  int channel;
  int64_t download_can_id;
  int64_t upload_can_id;
};

struct XcpData {
  std::string name;
  std::string value_type;
  std::vector<uint8_t> value_length;
  std::vector<uint8_t> dim;
  std::vector<uint8_t> address;
  std::vector<uint8_t> data;
};

// help function to print xcp data in Json
void PrintXcpData(std::vector<XcpData>* vec_data);

std::vector<float> GetFloat32FromBuffer(const std::vector<uint8_t>& buffer);
std::vector<uint8_t> StringToVectorUInt8(std::string str);
std::vector<float> StringToVectorFloat32(std::string str);
std::string VectorUInt8ToString(std::vector<uint8_t> buffer);
void PrintFloat32Vector(const std::vector<float>& vec_data);
bool IsSame(const std::vector<uint8_t>& lvalue,
            const std::vector<uint8_t>& rvalue);
}  // namespace xcp
}  // namespace newrizon

#endif  // SRC_INC_XCP_DATA_H_
