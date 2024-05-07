# PDPA-PAN

This is the official repository for “Large-Scale High Altitude UAV-based Vehicle Detection via Pyramid Dual Pooling Attention Path Aggregation Network”. The repo is based on [JDet](https://github.com/Jittor/JDet).

## Abstract

UAVs can collect vehicle data in high altitude scenes, playing a significant role in intelligent urban management due to their wide of view. Nevertheless, the current datasets for UAV-based vehicle detection are acquired at altitude below 150 meters. This contrasts with the data perspective obtained from high altitude scenes, potentially leading to incongruities in data distribution. Consequently, it is challenging to apply these datasets effectively in high altitude scenes, and there is an ongoing obstacle. To resolve this challenge, we developed a comprehensive high altitude dataset named LH-UAV-Vehicle, specifically collected at flight altitudes ranging from 250 to 400 meters. Collecting data at higher flight altitudes offers a broader perspective, but it concurrently introduces complexity and diversity in the background, which consequently impacts vehicle localization and recognition accuracy. In response, we proposed the pyramid dual pooling attention path aggregation network (PDPA-PAN), an innovative framework that improves detection performance in high altitude scenes by combining spatial and semantic information from distinct feature layers. Object attention integration in both spatial and channel dimensions is aimed by the pyramid dual pooling attention module (PDPAM), which is achieved through the parallel integration of two distinct attention mechanisms. Furthermore, we have individually developed the pyramid pooling attention module (PPAM) and the dual pooling attention module (DPAM). The PPAM emphasizes channel attention, while the DPAM prioritizes spatial attention. This design aims to enhance vehicle information and suppress background interference more effectively. Extensive experiments conducted on the LH-UAV-Vehicle dataset conclusively demonstrate the efficacy of the proposed vehicle detection method.

## Requirements
- Ubuntu 18.04
- Python 3.7
- clang >= 8.0
- g++ >= 5.4.0

## Installation

Please refer to [JDet](https://github.com/Jittor/JDet) for installation and data preprocessing. Also, you could refer to "requirements_for_install.txt".

## Getting Started

### Train or Test

Refer to "requirements_for_install.txt".

### Evaluation

We use [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit) to evaluate our model.

## LH-UAV-Vehicle

LH-UAV-Vehicle dataset is available. The following is the method to apply for a dataset：
1) Use the school's email address (edu, stu, etc.) and send an email to: yikuizhai@126.com
2) Sign the relevant agreement to ensure that it will only be used for scientific research and not for commercial purposes.A scanned version of the agreement that requires a handwritten signature. Both Chinese and English signatures are acceptable.
3) Authorization will be obtained in 1-3 days.
(Notice: If you use this dataset as the benchmark dataset for your paper, please cite the paper)

## Citation

Please cite this if you want to use it in your work.

```
@ARTICLE{jackychou,
  title={Large-Scale High Altitude UAV-based Vehicle Detection via Pyramid Dual Pooling Attention Path Aggregation Network}, 
  author={Ying, Zilu, and Zhou, Jianhong, and Zhai, Yikui Jianhong Zhou, Yikui Zhai and Quan, Hao and Li, Wenba and Genovese, Angelo and Piuri, Vincenzo and Fabio, Scotti},
  journal={IEEE Transactions on Intelligent Transportation System}, 
  year={2024},
  volume={-},
  number={-},
  pages={-},
  doi={10.1109/TITS.2024.3396915}
}
```
