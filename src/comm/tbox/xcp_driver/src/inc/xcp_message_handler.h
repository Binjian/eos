// Copyright 2020 Newrizon. All rights reserved.
// Use of this source code  can be found in the LICENSE file,
// which is part of this source code package.

#ifndef SRC_INC_XCP_MESSAGE_HANDLER_H_
#define SRC_INC_XCP_MESSAGE_HANDLER_H_

#include <vector>
#include <string>
#include <boost/thread.hpp>
#include "inc/xcp_data.h"
#include "inc/socket_can_driver_class.h"

namespace newrizon {
namespace xcp {

enum XcpErrorCode { XCP_SUCESS = 0, XCP_ERROR = 1, XCP_TIMEOUT = 2 };
enum XcpStauts { DISCONNECTED = 0, CONNECTED = 1, DOWNLOADING = 2 };

enum XcpCommandCode {
  CONNECT = 0xFF,
  DISCONNECT = 0xFE,
  SET_MTA = 0xF6,
  DOWNLOAD = 0xF0,
  SHORT_UPLOAD = 0xF4,
  UPLOAD = 0xF5,
  DOWNLOAD_NEXT = 0xEF
};

struct XcpCommand{
  uint8_t msg[8];
  uint8_t length;
};

class XCPMessageHandler {
 public:
  static XCPMessageHandler* GetInstance();
  void SetXcpInfo(const XcpInfo& xcp_info);
  void DownloadXcpData(const std::vector<XcpData>& vec_data);
  void UploadXcpData(std::vector<XcpData>* vec_data);

 private:
  XCPMessageHandler();
  static XCPMessageHandler* instance_;
  static XcpInfo xcp_info_;
  can_driver::SocketCanDriver* can_driver_;
  XcpStauts status_;
  int Connect();
  // int Download(uint8_t* address, uint8_t* msg, uint8_t message_length);
  int Download(const std::vector<uint8_t>& address,
               const std::vector<uint8_t>& download_message);
  int DownloadDataBlock(const std::vector<uint8_t>& download_message);
  std::vector<uint8_t> Upload(const std::vector<uint8_t>& address,
                              const uint32_t& message_length);
  std::vector<uint8_t> ShortUpload(const std::vector<uint8_t>& address,
                             const uint8_t& message_length);
  std::vector<uint8_t> LongUpload(const std::vector<uint8_t>& address,
                              const uint32_t& message_length);
  XcpErrorCode Disconnect();

  void SetMta(const std::vector<uint8_t>& address);
  // return
  //   0:sucess; -1:timeout
  XcpErrorCode WaitResponse(const std::string& hex_str, const int& timeout_ms);

  // void ReceiveCanMessage(int channel);
  boost::thread* can_receive_thread_;
};

}  // namespace xcp
}  // namespace newrizon

#endif  // SRC_INC_XCP_MESSAGE_HANDLER_H_
