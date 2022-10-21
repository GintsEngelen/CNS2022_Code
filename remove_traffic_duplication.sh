#!/bin/sh

if [ "$#" != "2" ]; then
    echo "Usage : $0 pcap_folder output_folder"
else
    # Make sure folders exist
    mkdir -p "$1"
    mkdir -p "$2"

    # Parameter
    timewindow=0.000500

    # Remove duplicated traffic with editcap
    find "$1" -iname "*.pcap" | while read line; do editcap -w "$timewindow" "$line" "$2/$(basename $line)"; done
fi