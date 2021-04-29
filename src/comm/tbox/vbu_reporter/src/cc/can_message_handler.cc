// Copyright 2020 Newrizon. All rights reserved.
// Use of this source code  can be found in the LICENSE file,
// which is part of this source code package.

#include "inc/can_message_handler.h"

#include <iostream>

#include "inc/can_driver_config.h"
#include "inc/can_message_processer.h"
#include "inc/socket_can_driver_class.h"
#include "inc/dbc_parser.h"

namespace newrizon {
namespace can_driver {

CanMessageHandler* CanMessageHandler::instance_ = NULL;

CanMessageHandler* CanMessageHandler::GetInstance() {
  if (instance_ == NULL) {
    instance_ = new CanMessageHandler();
  }
  return instance_;
}

CanMessageHandler::CanMessageHandler() {}

void CanMessageHandler::Start() {
  // start can receiving threads
  for (int i = 0; i < newrizon::config::can_driver_max_can_num; ++i) {
    if (newrizon::config::can_driver_channel_rx_in_used[i]) {
      can_receive_threads_[i] = new boost::thread{
          boost::bind(&CanMessageHandler::ReceiveCanMessage, this, i)};
    }
    if (newrizon::config::can_driver_channel_tx_in_used[i]) {
      can_publish_threads_[i] = new boost::thread{
          boost::bind(&CanMessageHandler::PublishCanMessage, this, i)};
    }
  }
}


void CanMessageHandler::ReceiveCanMessage(int channel) {
  int64_t id;
  unsigned char msg[8];
  unsigned int dlc;
  unsigned int flag;
  uint64_t time;
  do {
    SocketCanDriver::GetInstance()->ReadBlock(channel, &id, msg, &dlc, &flag,
                                              &time);
    DbcParser::GetInstance()->ParseMessage(channel, id, msg, dlc);
    CanMessageProcesser* can_message_processer =
        CanMessageProcesser::GetInstance();
    switch (channel) {
      case 0:
        can_message_processer->ProcessCan0Messages();
        break;
      case 1:
        can_message_processer->ProcessCan1Messages();
        break;
      case 2:
        can_message_processer->ProcessCan2Messages();
        break;
      case 3:
        can_message_processer->ProcessCan3Messages();
        break;
      default:
        break;
    }
  } while (true);
}

void CanMessageHandler::PublishCanMessage(int channel) {
  do {
    DbcParser::GetInstance()->TransmitMessages(channel);
    boost::this_thread::sleep_for(boost::chrono::milliseconds(
        config::can_driver_can_publish_interval[channel]));
  } while (true);
}

}  // namespace can_driver
}  // namespace newrizon
