#!/bin/bash

SCRIPT_DIR="$( cd -P "$( dirname "$BASH_SOURCE[0]" )" >/dev/null 2>&1 && pwd )"
cd $SCRIPT_DIR/../
ROOT_DIR=$PWD

echo "start building ..."
$SCRIPT_DIR/build.sh
echo "start running ..."

$SCRIPT_DIR/can_activate.sh
sleep 1
$ROOT_DIR/build/vbu_reporter_node

