#!/usr/bin/env python3

import argparse
import json
import struct
import subprocess

g_input_json_path = "../xcp_driver/json/example.json"
g_output_json_path = "../xcp_driver/json/download.json"
g_download_script = "../xcp_driver/scripts/download.sh"


def float_to_hex(value):
    h = hex(struct.unpack(">I", struct.pack("<f", value))[0])
    return h


def hex_to_float(value):
    return float(struct.unpack(">f", struct.pack("<I", value))[0])


def float_array_to_buffer(float_array):
    buffer_value = ""
    for i in range(len(float_array)):
        hex_str = float_to_hex(float_array[i])[2:]
        if len(hex_str) < 8:
            diff = 8 - len(hex_str)
            hex_str = "0" * diff + hex_str
        buffer_value = buffer_value + hex_str
    return buffer_value


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("example_json", help="example json file path")
    parser.add_argument(
        "-o", "--output", help="<Required> output json file name", required=True
    )
    args = parser.parse_args()
    return args


def write_json(output_json_path, example_json_path, data):
    # 1 read example json
    f = open(example_json_path, "r")
    json_obj = json.load(f)
    f.close()
    # 2 write values to json object
    for item in data:
        name = item["name"]
        value = item["value"]
        for i in range(len(json_obj["data"])):
            if json_obj["data"][i]["name"] == name:
                dim = json_obj["data"][i]["dim"]
                value_length = json_obj["data"][i]["value_length"]
                length = 1
                for d in dim:
                    length = length * d
                if len(value) != length * value_length * 2:
                    print("value length does not match")
                    return
                json_obj["data"][i]["value"] = value
    # 3 write output json
    f = open(output_json_path, "w")
    json_str = json.dumps(json_obj)
    f.write(json_str)
    f.close()


def send_float_array(name, float_array):
    value_str = float_array_to_buffer(float_array)
    data = [{"name": name, "value": value_str}]
    write_json(g_output_json_path, g_input_json_path, data)
    xcp_download = subprocess.run([g_download_script])
    print("The exit code was: %d" % xcp_download.returncode)


if __name__ == "__main__":
    # args = parse_arg()
    value = [99.0] * 21 * 17
    send_float_array("TQD_trqTrqSetECO_MAP_v", value)
