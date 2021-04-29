// Copyright 2020 Newrizon. All rights reserved.
// Use of this source code  can be found in the LICENSE file,
// which is part of this source code package.

#ifndef SRC_INC_DBC_PARSER_H_
#define SRC_INC_DBC_PARSER_H_

#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>

#include "dbc/dbc_node.h"
#include "inc/can_driver_config.h"

namespace newrizon {
namespace can_driver {


class DbcParser {
 public:
  static DbcParser* GetInstance();
  void ParseMessage(int channel, uint32_t id, const uint8_t* msg, uint16_t dlc);
  void TransmitMessages(int channel);

 private:
  DbcParser();
  void InitDbcLists();
  static DbcParser* instance_;
  static Dbc_node* dbc_parser_;
  static DbcTableList dbc_lists_[newrizon::config::can_driver_max_can_num];
  static DbcParserDbcTblType can0_dbc_list_[2];
  static DbcParserDbcTblType can1_dbc_list_[1];
  static DbcParserDbcTblType can2_dbc_list_[1];
  static DbcParserDbcTblType can3_dbc_list_[1];
};

}  // namespace can_driver
}  // namespace newrizon

#endif  // SRC_INC_DBC_PARSER_H_
