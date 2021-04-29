// Copyright 2020 Newrizon. All rights reserved.
// Use of this source code  XCP be found in the LICENSE file,
// which is part of this source code package.

#include "inc/xcp_message_handler.h"

#include <iostream>
#include <boost/chrono.hpp>

#include "inc/socket_can_driver_class.h"
#include "inc/xcp_driver_config.h"

namespace newrizon {
namespace xcp {

XCPMessageHandler* XCPMessageHandler::instance_ = NULL;

XcpInfo XCPMessageHandler::xcp_info_;

XCPMessageHandler* XCPMessageHandler::GetInstance() {
  if (instance_ == NULL) {
    instance_ = new XCPMessageHandler();
  }
  return instance_;
}

XCPMessageHandler::XCPMessageHandler() {
  can_driver_ = can_driver::SocketCanDriver::GetInstance();
}

int XCPMessageHandler::Connect() {
  uint8_t out_buff[] = {CONNECT, 0x00};
  uint32_t out_dlc = 2;
  can_driver_->Write(xcp_info_.channel, xcp_info_.download_can_id, out_buff,
                     out_dlc);
  int64_t id;
  unsigned char msg[8];
  unsigned int dlc;
  unsigned int flag;
  uint64_t time;
  // can_driver_->ReadBlock(xcp_info_.channel, &id, msg, &dlc, &flag, &time);

  // std::cout << "connecting : " << std::endl;
  // std::cout << "channel : " << xcp_info_.channel << std::endl;
  do {
    can_driver_->ReadBlock(xcp_info_.channel, &id, msg, &dlc, &flag, &time);
    if (id == xcp_info_.upload_can_id && msg[0] == 0xFF) {
      std::cout << "connecting : " << std::endl;
      for (int i = 0; i < dlc; i++) printf("%02X ", msg[i]);
      printf("\n");
      break;
    }
    // std::cout << "id : " << id << std::endl;
    // for (int i = 0; i < dlc; i++) printf("%02X ", msg[i]);
    // std::cout << std::endl;
  } while (true);
  // if (msg[0] == 0xFF) {
  std::cout << "connected" << std::endl;
  // }
  // for (int i = 0; i < dlc; i++) printf("%02X \n", msg[i]);
  // printf("%02X \n", msg[1]);
  // printf("%d \n", msg[1]);

  XcpErrorCode ret = XCP_SUCESS;
  return ret;
}

XcpErrorCode XCPMessageHandler::Disconnect() {
  uint8_t out_buff[] = {DISCONNECT};
  uint32_t out_dlc = 1;
  can_driver_->Write(xcp_info_.channel, xcp_info_.download_can_id, out_buff,
                     out_dlc);
  XcpErrorCode err = WaitResponse("ff", 5000);
  if (err == XCP_SUCESS) {
    std::cout << "disconnected" << std::endl;
    return XCP_SUCESS;
  } else {
    std::cout << "disconnected fail" << std::endl;
    return XCP_ERROR;
  }
}

int XCPMessageHandler::Download(const std::vector<uint8_t>& address,
                                const std::vector<uint8_t>& download_message) {
  SetMta(address);
  XcpErrorCode err = WaitResponse("ff", 5000);
  if (err == XCP_SUCESS) {
    std::cout << "SET_MTA Sucess" << std::endl;
  } else {
    std::cout << "SET_MTA Fail" << std::endl;
  }
  uint32_t total_length = download_message.size();
  int block_size = newrizon::config::xcp_max_download_block_size;
  int loop_time = total_length / block_size;
  int last_message_length = total_length % block_size;
  std::cout << "total_length : " << total_length << std::endl;
  std::cout << "last_message_length : " << last_message_length << std::endl;

  // boost::this_thread::sleep_for(
  //     boost::chrono::mircoseconds(newrizon::config::xcp_min_st));
  // std::vector<uint8_t> test_block;
  // std::vector<uint8_t> block(v1.begin() + 1, v1.end());
  for (int i = 0; i < loop_time; ++i) {
    std::cout << "loop : " <<  i << std::endl;
    std::cout << "start : " << block_size * i << std::endl;
    std::cout << "end : " << block_size * (i + 1) << std::endl;
    std::vector<uint8_t> block(download_message.begin() + block_size * i,
                               download_message.begin() + block_size * (i + 1));

    std::cout << "block size : " << block.size() << std::endl;
    // test_block.insert(test_block.end(), block.begin(), block.end());
    DownloadDataBlock(block);
  }
  // last message
  if (block_size * loop_time != total_length) {
    std::cout << "last message : " << std::endl;
    std::cout << "start : " << block_size * loop_time << std::endl;
    std::cout << "end : " << block_size * loop_time + last_message_length
              << std::endl;
    std::vector<uint8_t> block(
        download_message.begin() + block_size * loop_time,
        download_message.begin() + total_length);
    std::cout << "block size : " << block.size() << std::endl;
    DownloadDataBlock(block);
  }
  // if (IsSame(test_block, download_message)) {
  //   std::cout << "is same gooooood" << std::endl;
  // } else {
  //   std::cout << "fucked up" << std::endl;
  // }
}

int XCPMessageHandler::DownloadDataBlock(
    const std::vector<uint8_t>& download_message) {
  int i = 0;
  int j = 0;
  uint8_t download_message_length = download_message.size();
  int block_size = newrizon::config::xcp_max_download_block_size;
  if (download_message_length > block_size) {
    return -1;
  }

  int64_t id;
  unsigned char msg[8];
  unsigned int dlc;
  unsigned int flag;
  uint64_t time;

  int remaining_length = download_message_length;
  int transmit_length = 0;
  int loop_time = ceil(download_message_length / 6.0);
  // std::cout << "loop time : " << loop_time << std::endl;
  if (remaining_length > 6) {
    transmit_length = 6;
  } else {
    transmit_length = remaining_length;
  }
  uint8_t out_buff[8];
  // send first message
  out_buff[0] = DOWNLOAD;
  out_buff[1] = remaining_length;
  for (j = 0; j < transmit_length; j++) {
    out_buff[j + 2] = download_message[j];
  }
  // std::cout << "Write " << 1 << std::endl;
  // for (int i = 0; i < transmit_length + 2; i++) printf("%02X ", out_buff[i]);
  // printf("\n");
  can_driver_->Write(xcp_info_.channel, xcp_info_.download_can_id, out_buff,
                     transmit_length + 2);

  // send rest of the message if in block mode
  for (i = 1; i < loop_time; ++i) {
    // send single message
    remaining_length = remaining_length - 6;
    out_buff[0] = DOWNLOAD_NEXT;
    out_buff[1] = remaining_length;

    if (remaining_length > 6) {
      transmit_length = 6;
    } else {
      transmit_length = remaining_length;
    }
    for (j = 0; j < transmit_length; j++) {
      out_buff[j + 2] = download_message[6 * i + j];
    }
    // std::cout << "Write " << i + 1 << std::endl;
    // std::cout << "remaining length " << remaining_length << std::endl;
    // std::cout << "transmit length " << transmit_length << std::endl;
    // for (int j = 0; j < transmit_length + 2; j++) printf("%02X ",
    // out_buff[j]); printf("\n");
    boost::this_thread::sleep_for(
        boost::chrono::microseconds(newrizon::config::xcp_min_st));
    can_driver_->Write(xcp_info_.channel, xcp_info_.download_can_id, out_buff,
                       transmit_length + 2);
  }
  // out_buff[0] = DOWNLOAD_NEXT;
  // out_buff[1] = 0;
  // can_driver_->Write(xcp_info_.channel, xcp_info_.download_can_id, out_buff, 2);

  do {
    can_driver_->ReadBlock(xcp_info_.channel, &id, msg, &dlc, &flag, &time);
  } while (id != xcp_info_.upload_can_id || msg[0] != 0xFF);
  std::cout << "Write Sucess" << std::endl;

  XcpErrorCode ret = XCP_SUCESS;
  return ret;
}

std::vector<uint8_t> XCPMessageHandler::Upload(
    const std::vector<uint8_t>& address, const uint32_t& message_length) {
  std::vector<uint8_t> ret;
  if (message_length <= 7) {
    std::cout << "short uplaod" << std::endl;
    ret = ShortUpload(address, message_length);
  } else if (message_length > 7) {
    std::cout << "long uplaod" << std::endl;
    ret = LongUpload(address, message_length);
  }
  return ret;
}

std::vector<uint8_t> XCPMessageHandler::LongUpload(
    const std::vector<uint8_t>& address, const uint32_t& message_length) {
  int i = 0;
  int j = 0;
  int64_t id;
  unsigned char msg[8];
  unsigned int dlc;
  unsigned int flag;
  uint64_t time;

  std::vector<uint8_t> ret;

  std::cout << "SET_MTA start" << std::endl;
  SetMta(address);

  XcpErrorCode err = WaitResponse("ff", 5000);
  if (err == XCP_SUCESS) {
    std::cout << "SET_MTA Sucess" << std::endl;
  } else {
    std::cout << "SET_MTA Fail" << std::endl;
    return ret;
  }

  printf("message length : %d \n", message_length);
  uint8_t loop_time = message_length / 255;
  uint8_t last_message_length = message_length % 255;

  printf("loop_time : %d \n", loop_time);
  printf("last message length : %d \n", last_message_length);

  // loop n times
  for (int n = 0; n < loop_time; ++n) {
    // std::cout << "loop : " << n << std::endl;
    uint8_t out_buff[8];
    out_buff[0] = UPLOAD;
    out_buff[1] = 255;
    can_driver_->Write(xcp_info_.channel, xcp_info_.download_can_id, out_buff,
                       2);
    int received_bytes = 0;
    while (received_bytes < 255) {
      can_driver_->ReadBlock(xcp_info_.channel, &id, msg, &dlc, &flag, &time);
      if (id == xcp_info_.upload_can_id && msg[0] == 0xFF) {
        for (i = 1; i < dlc && received_bytes < 255; ++i) {
          // printf("%02X ", msg[i]);
          ret.push_back(msg[i]);
          received_bytes++;
        }
        // printf("\n");
      }
    }
    // std::cout << "ret length after loop : " << i << " :" << ret.size()
    //           << std::endl;
  }

  // get last message
  uint8_t out_buff[8];
  out_buff[0] = UPLOAD;
  out_buff[1] = last_message_length;
  can_driver_->Write(xcp_info_.channel, xcp_info_.download_can_id, out_buff, 2);
  int received_bytes = 0;
  // printf("last message \n");
  while (received_bytes < last_message_length) {
    // while (ret.size() < message_length) {
    can_driver_->ReadBlock(xcp_info_.channel, &id, msg, &dlc, &flag, &time);
    if (id == xcp_info_.upload_can_id && msg[0] == 0xFF) {
      for (i = 1; i < dlc && received_bytes < last_message_length; ++i) {
        // printf("%02X ", msg[i]);
        ret.push_back(msg[i]);
        received_bytes++;
      }
      // printf("\n");
    }
  }
  std::cout << "ret length : " << ret.size() << std::endl;
  return ret;
}

std::vector<uint8_t> XCPMessageHandler::ShortUpload(
    const std::vector<uint8_t>& address, const uint8_t& message_length) {
  std::vector<uint8_t> ret;
  uint8_t out_buff[8] = {SHORT_UPLOAD, message_length, 0x00,       0x00,
                         address[3],   address[2],     address[1], address[0]};
  int i;
  int64_t id;
  unsigned char msg[8];
  unsigned int dlc;
  unsigned int flag;
  uint64_t time;
  can_driver_->Write(xcp_info_.channel, xcp_info_.download_can_id, out_buff, 8);

  do {
    can_driver_->ReadBlock(xcp_info_.channel, &id, msg, &dlc, &flag, &time);
    if (id == xcp_info_.upload_can_id && msg[0] == 0xFF) {
      for (i = 1; i < dlc && ret.size() < message_length; ++i) {
        ret.push_back(msg[i]);
      }
      break;
    }
  } while (true);
  printf("upload data begin : \n");
  for (int i = 0; i < dlc; i++) printf("%02X ", msg[i]);
  printf("\n");
  return ret;
}

void XCPMessageHandler::SetXcpInfo(const XcpInfo& xcp_info) {
  // start XCP receiving threads
  xcp_info_ = xcp_info;
  status_ = DISCONNECTED;

  //  uint8_t address[4] = {0x44, 0xFC, 0x01, 0x70};
  //  Connect();
  //
  //  uint8_t message0[] = {0x00, 0x00, 0x48, 0x44};
  //  Download(address, message0, 4);
  //  ShortUpload(address, 4);
  //
  //  uint8_t message1[] = {0x01, 0x02, 0x03, 0x04};
  //  Download(address, message1, 4);
  //  ShortUpload(address, 4);
  //
  //  Download(address, message0, 4);
  //  ShortUpload(address, 4);
}

void XCPMessageHandler::DownloadXcpData(const std::vector<XcpData>& vec_data) {
  Connect();
  std::vector<XcpData>::const_iterator it;
  for (it = vec_data.begin(); it != vec_data.end(); ++it) {
    Download(it->address, it->data);
  }
  Disconnect();
}

void XCPMessageHandler::UploadXcpData(std::vector<XcpData>* vec_data) {
  Connect();
  std::vector<XcpData>::iterator it;
  for (it = vec_data->begin(); it != vec_data->end(); ++it) {
    uint32_t value_bytes = 1;
    uint32_t data_length = 1;
    if (it->value_type == "FLOAT32_IEEE") {
      value_bytes = 4; }
    for (int i = 0; i < it->dim.size(); i++) {
      data_length = data_length * it->dim[i];
    }
    uint32_t total_bytes = data_length * value_bytes;
    printf("data_length: %d \n", data_length);
    printf("value_bytes: %d \n", value_bytes);
    printf("total bytes: %d \n", total_bytes);

    auto uploaded_data = Upload(it->address, total_bytes);
    if (IsSame(it->data, uploaded_data)) {
      std::cout << "uploaded data is same with the json file" << std::endl;
    } else {
      it->data = uploaded_data;
    }
  }
  Disconnect();
  // PrintXcpData(vec_data);
}

XcpErrorCode XCPMessageHandler::WaitResponse(const std::string& hex_str,
                                             const int& timeout_ms) {
  int i;
  int64_t id;
  unsigned char msg[8];
  unsigned int dlc;
  unsigned int flag;
  uint64_t time;

  auto vec_hex = StringToVectorUInt8(hex_str);

  auto tick = boost::posix_time::second_clock::local_time();
  int delta_ms = 0;
  while (delta_ms < timeout_ms) {
    can_driver_->ReadBlock(xcp_info_.channel, &id, msg, &dlc, &flag, &time);
    auto now = boost::posix_time::second_clock::local_time();
    boost::posix_time::time_duration diff = tick - now;
    delta_ms = diff.total_milliseconds();
    // if match hex string
    if (dlc != vec_hex.size()) {
      continue;
    }
    for (i = 0; i < dlc; ++i) {
      if (msg[i] != vec_hex[i]) {
        continue;
      }
    }
    return XCP_SUCESS;
  }
  return XCP_ERROR;
}

void XCPMessageHandler::SetMta(const std::vector<uint8_t>& address) {
  uint8_t out_buff[8] = {SET_MTA,    0x00,       0x00,       0x00,
                         address[3], address[2], address[1], address[0]};
  can_driver_->Write(xcp_info_.channel, xcp_info_.download_can_id, out_buff, 8);
}

}  // namespace xcp
}  // namespace newrizon
