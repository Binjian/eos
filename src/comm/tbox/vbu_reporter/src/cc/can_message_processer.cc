// Copyright 2020 Newrizon. All rights reserved.
// Use of this source code  can be found in the LICENSE file,
// which is part of this source code package.

#include "inc/can_message_processer.h"
#include "inc/socket_can_driver_class.h"
#include "inc/dbc_parser.h"
#include "inc/http_message_handler.h"
#include "dbc/PTCAN.h"

namespace newrizon {
namespace can_driver {

CanMessageProcesser* CanMessageProcesser::instance_ = NULL;

CanMessageProcesser* CanMessageProcesser::GetInstance() {
  if (instance_ == NULL) {
    instance_ = new CanMessageProcesser();
  }
  return instance_;
}

CanMessageProcesser::CanMessageProcesser() {
}

void CanMessageProcesser::ProcessCan0Messages() {
}

void CanMessageProcesser::ProcessCan1Messages() {
  HttpOutputMsg out_msg;
  out_msg.vin = "L045A";
  out_msg.remaining_charging_time = VCU_RemainingChargingTime_501_CAN1;
  out_msg.remaining_driving_range = VCU_RemainingDrivingRange_501_CAN1;
  out_msg.bms_soc = VCU_BMSSOC_53_CAN1;
  out_msg.bms_display_soc = VCU_BMSDIsplaySOC_53_CAN1;
  HttpMessageHandler::WriteMessage(out_msg);
}

void CanMessageProcesser::ProcessCan2Messages() {}

void CanMessageProcesser::ProcessCan3Messages() {}

}  // namespace can_driver
}  // namespace newrizon
