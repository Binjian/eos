// Copyright 2020 Newrizon. All rights reserved.
// Use of this source code  can be found in the LICENSE file,
// which is part of this source code package.

#ifndef SRC_INC_CAN_MESSAGE_HANDLER_H_
#define SRC_INC_CAN_MESSAGE_HANDLER_H_

#include <boost/thread.hpp>
#include "inc/can_driver_config.h"

namespace newrizon {
namespace can_driver {

class CanMessageHandler {
 public:
  static CanMessageHandler* GetInstance();
  void Start();

 private:
  static CanMessageHandler* instance_;
  CanMessageHandler();
  void ReceiveCanMessage(int channel);
  void PublishCanMessage(int channel);
  boost::thread* can_receive_threads_[newrizon::config::can_driver_max_can_num];
  boost::thread* can_publish_threads_[newrizon::config::can_driver_max_can_num];
};

}  // namespace can_driver
}  // namespace newrizon

#endif  // SRC_INC_CAN_MESSAGE_HANDLER_H_
