# Error Prevalence in NIDS Datasets: A Case Study on CIC-IDS-2017 and CSE-CIC-IDS-2018

This repository contains the code used for our paper (Link to be added when proceedings are published). 
The code performs the labelling and benchmarking for the [CIC-IDS-2017](https://www.unb.ca/cic/datasets/ids-2017.html) 
and [CSE-CIC-IDS-2018](https://www.unb.ca/cic/datasets/ids-2018.html) datasets
 after it has been processed by [our modified version of the CICFlowMeter tool](https://github.com/GintsEngelen/CICFlowMeter). 

Note that all of this is *research code*.

If you use the code in this repository, please cite our paper:

            @inproceedings{liu2022error,
            title={Error Prevalence in NIDS datasets: A Case Study on CIC-IDS-2017 and CSE-CIC-IDS-2018},
            author={Liu, Lisa and Engelen, Gints and Lynar, Timothy and Essam, Daryl and Joosen, Wouter},
            booktitle={2022 IEEE Conference on Communications and Network Security (CNS)},
            pages={254--262},
            year={2022},
            organization={IEEE}
            }


An extended documentation of our paper can be found [here](https://intrusion-detection.distrinet-research.be/CNS2022/).

## How to use this repository

First, head over to the website of the dataset (either CIC-IDS-2017 or CSE-CIC-IDS-2018) and download 
the raw version of the dataset (PCAP file format). 

Then, navigate to "Original Network Traffic and Log data/Friday-02-03-2018/pcap" and delete the following file: 'capEC2AMAZ-O4EL3NG-172.31.69 - Copy.24' (This file contains traffic from the previous day and thus leads to duplicate flow entries).

Then, first run [pcapfix](https://github.com/Rup0rt/pcapfix) and then [reordercap](https://www.wireshark.org/docs/man-pages/reordercap.html)
on the PCAP files.

For CIC-IDS-2017 files, remove the duplicated traffic of the original pcap files provided by the authors using the provided script and the following syntax : `./remove_traffic_duplication.sh PCAP_folder_in PCAP_folder_out`.
Note that `PCAP_folder_in` should not be the same as `PCAP_folder_out`.

Then, run [our modified version of the CICFlowMeter tool](https://github.com/GintsEngelen/CICFlowMeter) on the data
obtained in the previous step:
 
1. Start the CICFlowMeter tool
2. Under the "NetWork" menu option, select "Offline"
3. Select the directory or directories containing the PCAP files. Note that for CSE-CIC-IDS-2018 you will have to run the
CICFlowMeter tool multiple times, once for each directory (where each directory corresponds to one day)
5. Keep the default values for the "Flow TimeOut" and "Activity Timeout" parameters (120000000 and 5000000 respectively)

This will generate the CSV files with the flows extracted from the raw PCAP files. 

For labelling of the CIC-IDS-2017 files, we used the CICIDS2017_labelling_fixed_CICFlowMeter.ipynb script. For labelling of the CSE-CIC-IDS-2018 files, we used the CICIDS2018_labelling_fixed_CICFlowMeter.ipynb script.

The two scripts with "original_version" in their name were used for our experiments where we determined the impact of 
just the labelling errors on classifiers. These scripts should only be used if you wish to reproduce our experimental results
as published in our paper.
