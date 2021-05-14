// Copyright 2020 Newrizon. All rights reserved.
// Use of this source code  can be found in the LICENSE file,
// which is part of this source code package.

#include "inc/json_reader.h"

#include <iostream>
#include <fstream>
#include <string>
#include <list>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/foreach.hpp>


namespace newrizon {
namespace xcp {

using std::cout;
using std::endl;
// Short alias for this namespace
namespace pt = boost::property_tree;

JsonReader::JsonReader() {
  vec_data_ = new std::vector<XcpData>;
  Init();
}

JsonReader::~JsonReader() {
  delete vec_data_;
}

void JsonReader::Init() {
  xcp_info_.channel = 0;
  xcp_info_.download_can_id = 0;
  xcp_info_.upload_can_id = 0;
  vec_data_->clear();
}

void JsonReader::ParseJson(const pt::ptree& root) {
  Init();
  // cout << "start reading json" << endl;
  // Load the json file in this ptree
  pt::ptree config = root.get_child("config");
  pt::ptree data = root.get_child("data");

  xcp_info_.channel = config.get<int>("channel");
  xcp_info_.download_can_id =
      std::stoul(config.get<std::string>("download_can_id"), nullptr, 16);
  xcp_info_.upload_can_id =
      std::stoul(config.get<std::string>("upload_can_id"), nullptr, 16);

  for (boost::property_tree::ptree::iterator it = data.begin();
       it != data.end(); ++it) {
    XcpData data;
    data.name = it->second.get<std::string>("name");
    std::string address = it->second.get<std::string>("address");
    data.address = StringToVectorUInt8(address);
    std::string value = it->second.get<std::string>("value");
    data.data = StringToVectorUInt8(value);
    std::string value_type = it->second.get<std::string>("value_type");
    data.value_type = value_type;
    pt::ptree pdim = it->second.get_child("dim");
    data.dim.clear();
    for (boost::property_tree::ptree::iterator it = pdim.begin();
         it != pdim.end(); ++it) {
      data.dim.push_back(it->second.get_value<uint8_t>());
    }
    vec_data_->push_back(data);
  }

  // cout << "end reading json" << endl;
}

void JsonReader::LoadJsonFromBuffer(const std::string& json) {
  std::stringstream ss;
  ss << json;
  pt::read_json(ss, root_);
  ParseJson(root_);
}

void JsonReader::LoadJsonFromPath(const std::string& file_path) {
  // Create a root
  pt::read_json(file_path, root_);
  ParseJson(root_);
}

void JsonReader::SaveJson(std::string path) {
  pt::ptree& data = root_.get_child("data");
  for (boost::property_tree::ptree::iterator it = data.begin();
       it != data.end(); ++it) {
    std::string name = it->second.get<std::string>("name");
    for (int i = 0; i < vec_data_->size(); ++i) {
      if (name == vec_data_->at(i).name) {
        std::string value_str = VectorUInt8ToString(vec_data_->at(i).data);
        // std::cout << "value_str length == " << value_str.size() << std::endl;
        it->second.put("value", value_str);
      } else {
      }
    }
  }
  pt::write_json(path, root_);
  // PrintXcpData(vec_data_);
}

}  // namespace xcp
}  // namespace newrizon
