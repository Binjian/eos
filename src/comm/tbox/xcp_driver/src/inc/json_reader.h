// Copyright 2020 Newrizon. All rights reserved.
// Use of this source code  can be found in the LICENSE file,
// which is part of this source code package.

#ifndef SRC_INC_JSON_READER_H_
#define SRC_INC_JSON_READER_H_

#include <string>
#include <vector>
#include <cstdint>
#include <boost/property_tree/ptree.hpp>
#include "inc/xcp_data.h"

namespace newrizon {
namespace xcp {

class JsonReader{
 public:
  JsonReader();
  ~JsonReader();
  void LoadJsonFromPath(const std::string& file_path);
  void LoadJsonFromBuffer(const std::string& json);
  void SaveJson(std::string path);
  XcpInfo GetXcpInfo() { return xcp_info_; }
  std::vector<XcpData>* GetData() { return vec_data_; }

 private:
  boost::property_tree::ptree root_;
  XcpInfo xcp_info_;
  std::vector<XcpData> *vec_data_;
  void Init();
  void ParseJson(const boost::property_tree::ptree& root);
};

}  // namespace xcp
}  // namespace newrizon

#endif  // SRC_INC_JSON_READER_H_
