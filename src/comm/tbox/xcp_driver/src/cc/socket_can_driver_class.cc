// Copyright 2020 Newrizon. All rights reserved.
// Use of this source code  can be found in the LICENSE file,
// which is part of this source code package.

#include "inc/socket_can_driver_class.h"

#include <iostream>
#include <string>
#include <vector>
#include "inc/xcp_driver_config.h"

namespace newrizon {
namespace can_driver {

SocketCanDriver* SocketCanDriver::instance_ = NULL;
CanHandlerInfo
    SocketCanDriver::can_info_[newrizon::config::can_driver_max_can_num];

SocketCanDriver* SocketCanDriver::GetInstance() {
  if (instance_ == NULL) {
    instance_ = new SocketCanDriver();
  }
  return instance_;
}

SocketCanDriver::SocketCanDriver() {
  can_info_[0] = {0, -1, CAN_ERR_RESERVED};
  can_info_[1] = {1, -1, CAN_ERR_RESERVED};
  can_info_[2] = {2, -1, CAN_ERR_RESERVED};
  can_info_[3] = {3, -1, CAN_ERR_RESERVED};
}

void SocketCanDriver::Start() {
  for (int i = 0; i < newrizon::config::can_driver_max_can_num; ++i) {
    if (newrizon::config::can_driver_channel_rx_in_used[i] ||
        newrizon::config::can_driver_channel_tx_in_used[i]) {
      instance_->OpenCan(&can_info_[i]);
    }
  }
}

void SocketCanDriver::OpenCan(CanHandlerInfo* can_info) {
  int s, i;
  struct sockaddr_can addr;
  struct ifreq ifr;

  if ((s = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0) {
    perror("Socket");
    return;
  }

  std::string interface_name = "can" + std::to_string(can_info->channel);
  // std::cout << interface_name << std::endl;
  memcpy(ifr.ifr_name, interface_name.c_str(), interface_name.length()+1);
  ioctl(s, SIOCGIFINDEX, &ifr);

  memset(&addr, 0, sizeof(addr));
  addr.can_family = AF_CAN;
  addr.can_ifindex = ifr.ifr_ifindex;

  if (bind(s, (struct sockaddr*)&addr, sizeof(addr)) < -1) {
    perror("Bind");
    return;
  }
  can_info->handler = s;
}

void SocketCanDriver::CloseCan(CanHandlerInfo* can_info) {
  if (close(can_info->handler) < 0) {
    perror("Close");
    return;
  }
}

SocketCanDriver::~SocketCanDriver() {
  for (int i = 0; i < newrizon::config::can_driver_max_can_num; ++i) {
    if (newrizon::config::can_driver_channel_rx_in_used[i] ||
        newrizon::config::can_driver_channel_tx_in_used[i]) {
      CloseCan(&can_info_[i]);
    }
  }
}

int SocketCanDriver::Write(uint16_t channel, uint32_t id, uint8_t* msg,
                           uint16_t dlc) {
  struct can_frame frame;
  frame.can_id = id;
  frame.can_dlc = dlc;
  memcpy(frame.data, msg, dlc);

  if (write(can_info_[channel].handler, &frame, sizeof(struct can_frame)) !=
      sizeof(struct can_frame)) {
    perror("Write");
    return 1;
  }

  return 0;
}

void SocketCanDriver::SetFilter(int channel, std::vector<uint32_t> bypass_ids) {
  int i;
  int id_num =  bypass_ids.size();
  uint64_t filter_size = sizeof(can_filter) * id_num;
  can_filter* filter = reinterpret_cast<can_filter*>(malloc(filter_size));

  for (i = 0; i < id_num; ++i) {
    filter[i].can_id = bypass_ids[i];
    filter[i].can_mask = 0xFFFF;
  }
  // struct can_filter rfilter[2];

  // rfilter[0].can_id = 0x630;
  // rfilter[0].can_mask = 0xFFFF;
  // rfilter[1].can_id = 0x631;
  // rfilter[1].can_mask = 0xFFFF;
  setsockopt(can_info_[channel].handler, SOL_CAN_RAW, CAN_RAW_FILTER, filter,
             filter_size);
}

void SocketCanDriver::ReadBlock(uint16_t channel, int64_t* id, uint8_t* msg,
                                uint32_t* dlc, uint32_t* flag, uint64_t* time) {
  int nbytes;
  struct can_frame frame;
  nbytes = read(can_info_[channel].handler, &frame, sizeof(struct can_frame));

  if (nbytes < 0) {
    perror("Read");
    return;
  }
  *id = frame.can_id;
  *dlc = frame.can_dlc;
  memcpy(msg, frame.data, frame.can_dlc);
  // printf("0x%03X [%d] ",frame.can_id, frame.can_dlc);
  // for (int i = 0; i < frame.can_dlc; i++)
  // printf("%02X ",frame.data[i]);

  // printf("\r\n");
}

}  // namespace can_driver
}  // namespace newrizon
