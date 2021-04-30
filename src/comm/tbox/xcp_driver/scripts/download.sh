#!/bin/bash

SCRIPT_DIR="$( cd -P "$( dirname "$BASH_SOURCE[0]" )" >/dev/null 2>&1 && pwd )"
cd $SCRIPT_DIR/../
ROOT_DIR=$PWD

#echo "start building ..."
#$SCRIPT_DIR/build.sh
#echo "start running ..."

$SCRIPT_DIR/can_activate.sh
sleep 1
cd $ROOT_DIR/build/
#./xcp_driver_node --mode download --input ../json/example.json --output test.json
./xcp_driver_node --mode download --input ../json/download.json --output test.json

