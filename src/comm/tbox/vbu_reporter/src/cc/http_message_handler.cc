// Copyright 2020 Newrizon. All rights reserved.
// Use of this source code  can be found in the LICENSE file,
// which is part of this source code package.

#include "inc/http_message_handler.h"

#include <curl/curl.h>

#include <iostream>
#include <istream>
#include <ostream>
#include <sstream>
#include <string>

#include "inc/can_driver_config.h"

namespace newrizon {
namespace can_driver {

HttpMessageHandler* HttpMessageHandler::instance_ = NULL;
HttpOutputMsg HttpMessageHandler::message_out_;
boost::mutex HttpMessageHandler::mutex_out_;
boost::thread* HttpMessageHandler::thread_;

HttpMessageHandler* HttpMessageHandler::GetInstance() {
  if (instance_ == NULL) {
    instance_ = new HttpMessageHandler();
  }
  return instance_;
}

HttpMessageHandler::HttpMessageHandler() {
  thread_ = new boost::thread{
      boost::bind(&HttpMessageHandler::ReportHttpMessage, this)};
}

void HttpMessageHandler::WriteMessage(const HttpOutputMsg& out_msg) {
  mutex_out_.lock();
  message_out_.vin = out_msg.vin;
  message_out_.remaining_charging_time = out_msg.remaining_charging_time;
  message_out_.remaining_driving_range = out_msg.remaining_driving_range;
  message_out_.bms_soc = out_msg.bms_soc;
  message_out_.bms_display_soc = out_msg.bms_display_soc;
  mutex_out_.unlock();
}

void HttpMessageHandler::ReportHttpMessage() {
  while (1) {
    mutex_out_.lock();
    HttpOutputMsg msg = message_out_;
    mutex_out_.unlock();
    // fill http message
    std::string jsonstr =
        "{\"vin\":\""
        + msg.vin
        + "\",\"remaining_charging_time\":\""
        + std::to_string(msg.remaining_charging_time)
        + "\",\"remaining_driving_range\":\""
        + std::to_string(msg.remaining_driving_range)
        + "\",\"bms_soc\":\""
        + std::to_string(msg.bms_soc)
        + "\",\"bms_display_soc\":\""
        + std::to_string(msg.bms_display_soc)
        + "\"}";
    std::cout << jsonstr << std::endl;

    CURL* curl;
    CURLcode res;

    // In windows, this will init the winsock stuff
    curl_global_init(CURL_GLOBAL_ALL);

    // get a curl handle
    curl = curl_easy_init();
    struct curl_slist* slist1;
    slist1 = NULL;
    slist1 = curl_slist_append(slist1, "Content-Type: application/json");

    if (curl) {
      // First set the URL that is about to receive our POST. This URL can
      // just as well be a https:// URL if that is what should receive the
      // data.
      curl_easy_setopt(curl, CURLOPT_HTTPHEADER, slist1);
      curl_easy_setopt(curl, CURLOPT_URL, config::bms_report_url);
      // Now specify the POST data
      curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonstr.c_str());

      // Perform the request, res will get the return code
      res = curl_easy_perform(curl);
      // Check for errors
      if (res != CURLE_OK)
        fprintf(stderr, "curl_easy_perform() failed: %s\n",
                curl_easy_strerror(res));

      // always cleanup
      curl_easy_cleanup(curl);
    }
    curl_global_cleanup();
    curl_slist_free_all(slist1);

    slist1 = NULL;

    boost::this_thread::sleep_for(
        boost::chrono::milliseconds(config::http_post_interval));
  }
}

}  // namespace can_driver
}  // namespace newrizon
