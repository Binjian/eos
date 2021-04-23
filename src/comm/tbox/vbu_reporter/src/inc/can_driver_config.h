// Copyright 2020 Newrizon. All rights reserved.
// Use of this source code  can be found in the LICENSE file,
// which is part of this source code package.

#ifndef SRC_INC_CAN_DRIVER_CONFIG_H_
#define SRC_INC_CAN_DRIVER_CONFIG_H_

namespace newrizon {
namespace config {
const int can_driver_max_can_num = 4;
const bool can_driver_channel_rx_in_used[4] = {false, true, false, false};
const bool can_driver_channel_tx_in_used[4] = {false, false, false, false};
const int http_post_interval = 10000;  // 1000 ms
const int can_driver_can_publish_interval[4] = {50, 50, 50, 50};  // 50 ms
const char bms_report_url[] =
     "http://tsp.newrizon.cloud/app/vehicle/energy_consumption";
//     "http://rap2.newrizon.work:38080/app/mock/17/app/vehicle/"
//     "energy_consumption";
}  // namespace config
}  // namespace newrizon

#endif  // SRC_INC_CAN_DRIVER_CONFIG_H_
