// Copyright 2020 Newrizon. All rights reserved.
// Use of this source code  can be found in the LICENSE file,
// which is part of this source code package.

#ifndef SRC_INC_SOCKET_CAN_DRIVER_CLASS_H_
#define SRC_INC_SOCKET_CAN_DRIVER_CLASS_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>

#include <linux/can.h>
#include <linux/can/raw.h>
#include <boost/chrono.hpp>
#include <boost/thread.hpp>

#include "dbc/dbc_node.h"

namespace newrizon {
namespace can_driver {

typedef enum {
  CAN_OK = 0,
  CAN_ERR_RESERVED = -99  ///< Reserved
} CanStatus;

struct CanHandlerInfo {
  int channel;
  int handler;
  CanStatus status;
};

class SocketCanDriver {
 public:
  static SocketCanDriver* GetInstance();
  static void Start();
  int Write(uint16_t channel, uint32_t canId, uint8_t* buff,
                        uint16_t dlc);
  void ReadBlock(uint16_t channel, int64_t* id, uint8_t* msg,
                       uint32_t* dlc, uint32_t* flag, uint64_t* time);

 private:
  static SocketCanDriver* instance_;
  static CanHandlerInfo can_info_[4];
  SocketCanDriver();
  ~SocketCanDriver();
  void OpenCan(CanHandlerInfo* can_info);
  void CloseCan(CanHandlerInfo* can_info);
};

}  // namespace can_driver
}  // namespace newrizon

#endif  // SRC_INC_SOCKET_CAN_DRIVER_CLASS_H_
