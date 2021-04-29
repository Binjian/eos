[img_design]: docs/images/VbuReporter.png

# Vbu Reporter

## Description

This node receive bms messages from VBU via CAN, and send those messages to
remote server by http post request in JSON format.

![img_design]

## Dependencies

* boost thread
* libcurl

## Usage

Tested on Ubuntu ( tested on 18.04 )

```bash
./scripts/run.sh
```
