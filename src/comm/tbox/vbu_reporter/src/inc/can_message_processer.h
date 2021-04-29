// Copyright 2020 Newrizon. All rights reserved.
// Use of this source code  can be found in the LICENSE file,
// which is part of this source code package.

#ifndef SRC_INC_CAN_MESSAGE_PROCESSER_H_
#define SRC_INC_CAN_MESSAGE_PROCESSER_H_

#include "dbc/dbc_node.h"
#include "inc/can_driver_config.h"
#include <boost/thread.hpp>

namespace newrizon {
namespace can_driver {

class CanMessageProcesser {
 public:
  static CanMessageProcesser* GetInstance();
  void ProcessCan0Messages();
  void ProcessCan1Messages();
  void ProcessCan2Messages();
  void ProcessCan3Messages();
 private:
  static CanMessageProcesser* instance_;
  CanMessageProcesser();
};

}  // namespace can_driver
}  // namespace newrizon

#endif  // SRC_INC_CAN_MESSAGE_PROCESSER_H_
