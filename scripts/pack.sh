#!/bin/bash
if [ -z "$1" ]; then
  echo "Usage: $0 <prefix>"
  exit 1
fi
apt install -y zip
zip -r ${1}-output.zip output