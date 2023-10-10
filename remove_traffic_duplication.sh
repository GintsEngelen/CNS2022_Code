#!/bin/bash

# The intention of this script is to remove unintended packet duplication from the 2017 pcaps.
# The mac_address and ip filtering to the impact of deduplication to known mac addresses and ip addresses, that have
# been confirmed to have definite duplication by prior analysis.
# The script works as follows
# 1. It takes a folder of pcaps as input, and an empty folder as output
# 2. It will filter out traffic from mac addresses and ip addresses that are known to be affected by packet duplication
# 3. It will then run editcap on the filtered pcaps to remove duplicated packets
# 4. It will then merge the original pcaps with the newly deduplicated pcaps and place them in the output folder


# Adjust this according to your system - number of jobs to run in parallel
max_jobs=5

if [ "$#" != "2" ]; then
    echo "Usage : $0 pcap_folder output_folder. pcap_folder should not be the same as output_folder"
else
    # Make sure folders exist
    mkdir -p "$1"
    mkdir -p "$2"

    temp_folder_dedup="$2/TempDeduplicated"
    temp_folder_orig="$2/TempOriginal"
    mkdir -p "$temp_folder_dedup"
    mkdir -p "$temp_folder_orig"

    # Parameter
    timewindow=0.000500
    # MAC addresses array
    mac_list=("00:c1:b1:14:eb:31" "01:00:0c:cc:cc:cc" "01:00:5e:00:00:16" "24:6e:96:4a:37:7a" "01:80:c2:00:00:0e")

    # Prepare for tshark display filter
    mac_filter_no="not ("

    for i in "${mac_list[@]}"
    do
        mac_filter_no+="eth.src == $i || "
    done

    mac_filter_no=${mac_filter_no::-4} # delete the last " && "

    mac_filter_no+=") && (eth.dst == ff:ff:ff:ff:ff:ff || ((ip.src == 192.168.0.0/16 || ip.src == 224.0.0.0/4) && (ip.dst == 192.168.0.0/16 || ip.dst == 224.0.0.0/4)))"
    mac_filter_yes="!(${mac_filter_no})"

    # Find pcap files
    find "$1" -iname "*.pcap" | while read line
    do
        # Filter by MAC address and place resultant pcap files in the temp folders
        (tshark -r "$line" -Y "${mac_filter_yes}" -w ${temp_folder_orig}/`basename "$line"`
        tshark -r "$line" -Y "${mac_filter_no}" -w ${temp_folder_dedup}/`basename "$line"`
        # Remove duplicated traffic with editcap in dedup folder
        editcap -w "$timewindow" ${temp_folder_dedup}/`basename "$line"` ${temp_folder_dedup}/temp_`basename "$line"`

        # Merge de-duplicated and original pcaps and move to main output folder
        mergecap -w "$2/$(basename $line)" ${temp_folder_dedup}/temp_`basename "$line"` ${temp_folder_orig}/`basename "$line"` ) &

      while (( $(jobs | wc -l) >= max_jobs )); do
        sleep 1
      done

    done

    # Wait for remaining jobs to finish before exiting
    wait
fi
