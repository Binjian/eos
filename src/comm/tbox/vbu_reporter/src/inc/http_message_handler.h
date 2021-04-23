// Copyright 2020 Newrizon. All rights reserved.
// Use of this source code  can be found in the LICENSE file,
// which is part of this source code package.

#ifndef SRC_INC_HTTP_MESSAGE_HANDLER_H_
#define SRC_INC_HTTP_MESSAGE_HANDLER_H_

#include <iostream>
#include <string>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>

namespace newrizon {
namespace can_driver {

struct HttpOutputMsg {
  std::string vin;
  float remaining_charging_time;
  float remaining_driving_range;
  float bms_soc;
  float bms_display_soc;
};

class HttpMessageHandler {
 public:
  static HttpMessageHandler* GetInstance();
  static void WriteMessage(const HttpOutputMsg& out_msg);

 private:
  HttpMessageHandler();
  void ReportHttpMessage();
  static HttpMessageHandler* instance_;
  static HttpOutputMsg message_out_;
  static boost::mutex mutex_out_;
  static boost::thread* thread_;
};
}  // namespace can_driver
}  // namespace newrizon

#endif  // SRC_INC_HTTP_MESSAGE_HANDLER_H_
