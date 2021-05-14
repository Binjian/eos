// Copyright 2020 Newrizon. All rights reserved.
// Use of this source code  can be found in the LICENSE file,
// which is part of this source code package.

#include <iostream>
#include <iterator>

#include <boost/program_options.hpp>
#include <boost/thread.hpp>

#include "inc/json_reader.h"
#include "inc/socket_can_driver_class.h"
#include "inc/xcp_data.h"
#include "inc/xcp_driver_config.h"
#include "inc/xcp_message_handler.h"

using std::string;
using std::cout;
using std::endl;
using std::exception;
using std::cerr;
using newrizon::xcp::XCPMessageHandler;
namespace po = boost::program_options;

po::variables_map ParseArgs(int argc, char** argv) {
  po::variables_map vm;
  try {
    po::options_description desc("Allowed options");
    desc.add_options()("help", "produce help message")(
        "mode", po::value<string>(), "set mode download or upload")(
        "input", po::value<string>(), "set input path")(
        "output", po::value<string>(), "set output path");

    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
      // cout << desc << "\n";
      exit(0);
    }

    if (vm.count("mode")) {
      string mode = vm["mode"].as<string>();
      // cout << "mode was set to " << mode << ".\n";
      if (mode == "upload") {
        if (vm.count("output")) {
          // cout << "output json : " << vm["output"].as<string>() << ".\n";
        } else {
          // cout << "output json path was not set.\n";
          exit(0);
        }
      }
    } else {
      // cout << "mode was not set.\n";
      exit(0);
    }

    if (vm.count("input")) {
      // cout << "input json : " << vm["input"].as<string>() << ".\n";
    } else {
      // cout << "input json path was not set.\n";
      exit(0);
    }

    return vm;
  } catch (exception& e) {
    cerr << "error: " << e.what() << "\n";
    exit(0);
  } catch (...) {
    cerr << "Exception of unknown type!\n";
    exit(0);
  }
}

int main(int argc, char** argv) {
  po::variables_map vm = ParseArgs(argc, argv);

  std::string mode = vm["mode"].as<string>();
  std::string input_json_path = vm["input"].as<string>();
  std::string output_json_path = vm["output"].as<string>();

  newrizon::xcp::JsonReader json_reader;
  json_reader.LoadJsonFromPath(input_json_path);
  // PrintXcpData(json_reader.GetData());

  newrizon::xcp::XcpInfo info = json_reader.GetXcpInfo();
  std::vector<uint32_t> bypass_ids;
  bypass_ids.push_back(info.upload_can_id);

  newrizon::can_driver::SocketCanDriver::GetInstance()->SetFilter(info.channel,
                                                                  bypass_ids);
  newrizon::can_driver::SocketCanDriver::GetInstance()->Start();

  XCPMessageHandler* xcp_handler = XCPMessageHandler::GetInstance();
  xcp_handler->SetXcpInfo(json_reader.GetXcpInfo());
  if (mode == "download") {
    // std::cout << "download" << std::endl;
    xcp_handler->DownloadXcpData(*json_reader.GetData());
  } else if (mode == "upload") {
    // std::cout << "upload" << std::endl;
    xcp_handler->UploadXcpData(json_reader.GetData());
    // std::cout << "save to json file : " << output_json_path << std::endl;
    json_reader.SaveJson(output_json_path);
  } else {
  }

  // while (1) {
  //   boost::this_thread::sleep_for(boost::chrono::milliseconds(
  //       newrizon::config::can_driver_main_thread_interval));
  // }

  return 0;
}
