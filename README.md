# Noise-control-multi-object-tracking
A plug-and-play solution mainly focuses on the noise-control designs of detection, association, and reconnection.

## Installations

Python3<br>
Pytorch 1.4+ or higher

## Module
### Heatmap Feature Prior Denoiser
[HFPD](https://github.com/Autoyou/Noise-control-multi-object-tracking/blob/main/HFPD.py) is applied to the features after the addition of heatmap and image fusion, such as the DLASeg part of dla.py in CenterTrack to highlight the central point features of heatmap.
### Smoothing Gain Kalman filter
[SG Kalman](https://github.com/Autoyou/Noise-control-multi-object-tracking/blob/main/kalman_filter.py) is applied to the Kalman filtering algorithm of data association, and it takes effect when the confidence is passed into the function project.
### Gradient Boosting Reconnection Context
[GBRC](https://github.com/Autoyou/Noise-control-multi-object-tracking/tree/main/GBRC) is used to reconnect the debris track after obtaining the result. Put the result into the [tmp](https://github.com/Autoyou/Noise-control-multi-object-tracking/tree/main/GBRC/tmp) folder, run [GBRC.main](https://github.com/Autoyou/Noise-control-multi-object-tracking/blob/main/GBRC/GBRC_main.py), and obtain the reconnection result in the [interresult](https://github.com/Autoyou/Noise-control-multi-object-tracking/tree/main/GBRC/interresult) folder. You need to change the name of the new file in opts.
# Citation

@article{you2023NCT,<br>
title={NCT:noise-control multi-object tracking},<br>
author={Zeng, Kai and You, Yujie and Shen, Tao and Wang, Qingwang and Tao, Zhimin and Wang, Zhifeng and Liu, Quanjun},<br>
journal = {Complex \& Intelligent Systems},<br>
year = {2023},<br>
doi={https://doi.org/10.1007/s40747-022-00946-9 }<br>
}
