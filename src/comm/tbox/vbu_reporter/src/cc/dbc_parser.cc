// Copyright 2020 Newrizon. All rights reserved.
// Use of this source code  can be found in the LICENSE file,
// which is part of this source code package.

#include "dbc/PTCAN.h"
#include "inc/dbc_parser.h"
#include "inc/socket_can_driver_class.h"

#include "dbc/dbc_node.h"

namespace newrizon {
namespace can_driver {

void loss_comm_handler(uint32_t can_id) {}

DbcParser* DbcParser::instance_ = NULL;
Dbc_node* DbcParser::dbc_parser_ = NULL;

DbcTableList DbcParser::dbc_lists_[newrizon::config::can_driver_max_can_num];
DbcParserDbcTblType DbcParser::can0_dbc_list_[2];
DbcParserDbcTblType DbcParser::can1_dbc_list_[1];
DbcParserDbcTblType DbcParser::can2_dbc_list_[1];
DbcParserDbcTblType DbcParser::can3_dbc_list_[1];

DbcParser* DbcParser::GetInstance() {
  if (instance_ == NULL) {
    instance_ = new DbcParser();
  }
  return instance_;
}

DbcParser::DbcParser() {
  InitDbcLists();
  dbc_parser_ = new Dbc_node(uint16_t(50), dbc_lists_,
                             newrizon::config::can_driver_max_can_num);

  dbc_parser_->setRxStatus(true);
  dbc_parser_->setTxStatus(true);
  dbc_parser_->setLossCommHandler(loss_comm_handler);
  std::function<int(uint16_t, uint32_t, uint8_t*, uint16_t)> WriteFunc(
      std::bind(&SocketCanDriver::Write, SocketCanDriver::GetInstance(),
                std::placeholders::_1, std::placeholders::_2,
                std::placeholders::_3, std::placeholders::_4));
  dbc_parser_->setTransmitHandler(WriteFunc);
  dbc_parser_->setLossCommCheckEnable(true);
}

void DbcParser::InitDbcLists() {
  // initialize message parser
  // Init can0 dbc
  dbc_lists_[0].dbc_list_ = can0_dbc_list_;
  dbc_lists_[0].size_ = 0;

  // Init can1 dbc
  can1_dbc_list_[0].pt_MsgTbl = TBL_DP_DBCMSGLIST_PTCAN;
  can1_dbc_list_[0].u16_tblSize = u16s_dp_MsgTblSize_PTCAN;
  dbc_lists_[1].dbc_list_ = can1_dbc_list_;
  dbc_lists_[1].size_ = 1;

  // Init can2 dbc
  dbc_lists_[2].dbc_list_ = can2_dbc_list_;
  dbc_lists_[2].size_ = 0;

  // Init can3 dbc
  dbc_lists_[3].dbc_list_ = can3_dbc_list_;
  dbc_lists_[3].size_ = 0;
}

void DbcParser::ParseMessage(int channel, uint32_t id, const uint8_t* msg,
                             uint16_t dlc) {
  dbc_parser_->rxIndication(id, msg, dlc, dbc_lists_[channel].dbc_list_,
                            dbc_lists_[channel].size_);
  dbc_parser_->parseMsg(dbc_lists_[channel].dbc_list_,
                        dbc_lists_[channel].size_);
}

void DbcParser::TransmitMessages(int channel) {
  dbc_parser_->transmitMessages(channel, dbc_lists_[channel].dbc_list_,
                                dbc_lists_[channel].size_);
}

}  // namespace can_driver
}  // namespace newrizon
