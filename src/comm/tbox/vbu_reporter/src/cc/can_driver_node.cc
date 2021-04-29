// Copyright 2020 Newrizon. All rights reserved.
// Use of this source code  can be found in the LICENSE file,
// which is part of this source code package.

#include "inc/socket_can_driver_class.h"
#include "inc/dbc_parser.h"
#include "inc/can_message_handler.h"
#include "inc/http_message_handler.h"
#include "inc/can_driver_config.h"

int main(int argc, char **argv) {
  newrizon::can_driver::SocketCanDriver::GetInstance()->Start();
  newrizon::can_driver::CanMessageHandler::GetInstance()->Start();
  newrizon::can_driver::HttpMessageHandler::GetInstance();

  while (1) {
    boost::this_thread::sleep_for(boost::chrono::milliseconds(
        newrizon::config::http_post_interval));
  }

  return 0;
}
