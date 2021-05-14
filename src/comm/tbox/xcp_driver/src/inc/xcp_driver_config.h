// Copyright 2020 Newrizon. All rights reserved.
// Use of this source code  can be found in the LICENSE file,
// which is part of this source code package.

#ifndef SRC_INC_XCP_DRIVER_CONFIG_H_
#define SRC_INC_XCP_DRIVER_CONFIG_H_

#include <cstdint>

namespace newrizon {
namespace config {
const int can_driver_max_can_num = 4;  // 20 Hz
const bool can_driver_channel_rx_in_used[4] = {false, false, false, true};
const bool can_driver_channel_tx_in_used[4] = {false, false, false, true};
const int can_driver_main_thread_interval = 100;  // ms
const uint8_t xcp_max_download_block_size = 6;
const uint8_t xcp_min_st = 200;  // min is 100 micro second
}  // namespace config
}  // namespace newrizon

#endif  // SRC_INC_XCP_DRIVER_CONFIG_H_
