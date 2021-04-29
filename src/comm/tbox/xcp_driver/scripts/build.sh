#!/bin/bash

SCRIPT_DIR="$( cd -P "$( dirname "$BASH_SOURCE[0]" )" >/dev/null 2>&1 && pwd )"
cd $SCRIPT_DIR/../
ROOT_DIR=$PWD

if [ -d "$ROOT_DIR/build" ]; then
  # Take action if $DIR exists. #
  cd $ROOT_DIR/build
else
  cd $ROOT_DIR
  mkdir build
  cd build
fi

cmake ..
make -j4
