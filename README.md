# Label-Decoupling-Module
This is the official PyTorch implementation of our work [Label Decoupling Strategy for 12-Lead ECG Classification](https://www.sciencedirect.com/science/article/pii/S0950705123000485), which has published in Knowledge-Based Systems. This repo contains some key codes of our LDM and its application in PTB-XL dataset.<br>
<div align=center>
<img width="850" src="https://github.com/Zhangshuojackpot/Label-Decoupling-Module/blob/main/introduction.png"/>
</div>

### Abstract
Automatic 12-lead ECG classification is always in the spotlight and has achieved great progress owing to the application of deep learning. After reviewing previous works, we find that most multi-label ECG models can be uniformly characterized as a deep embedding method. This method employs only one embedding space to produce high-dimensional features for all diagnostic labels, which results in a label entanglement phenomenon and produces at least three defects. The most serious one is that as the
number of labels goes up, the complexity of clustering in the penultimate layer grows exponentially. Motivated by these flaws, we provide an intuitive insight and propose a label decoupling module. It can solve these defects by launching the samples of various labels into multiple spaces. Then, we make a trainable mergence that combines the benefits of label decoupling and traditional label fusion to get the final prediction. In addition, we also introduce some metric learning techniques and further develop its large-margin version. It is important to note that our method is universal and can be applied with many state-of-the-art (SOTA) backbones for better performance in ECG classification. Experiments on benchmark datasets demonstrate that our approach strengthens all tested backbones and achieves better performance than various SOTA techniques in this field. 

### Preparation
The experimental environment is in [requirements.txt](https://github.com/Zhangshuojackpot/Label-Decoupling-Module/blob/main/requirements.txt).<br>

### Usage
1. Run [get_ptbxl.sh](https://github.com/Zhangshuojackpot/Label-Decoupling-Module/blob/main/get_ptbxl.sh) to download the PTB-XL dataset:<br>
```
./get_ptbxl.sh
```
2. Reproduce experimetal results:<br>
```
mkdir outputs
cd ./code_used_upload
python order_results_ptbxl1_upload.py
```

### Results
|Method|Macro-AUC(All)|Macro-AUC(Diag.)|Macro-AUC(Sub-diag.)|Macro-AUC(Super-diag.)|Macro-AUC(Form)|Macro-AUC(Rhythm)|Macro-AUC(Average)|
|:---|:---|:---|:---|:---|:---|:---|:---|
|LSTM|0.909(.002)|0.926(.002)|0.926(.002)|0.926(.000)|0.848(.006)|0.947(.003)|0.913(.003)|
|LSTM+LDM|0.923(.001)|0.931(.002)|0.935(.003)|0.930(.000)|0.851(.006)|0.948(.001)|0.920(.002)|
|Inception1d|0.925(.002)|0.928(.000)|0.927(.000)|0.918(.003)|0.886(.006)|0.948(.003)|0.922(.002)|
|Inception1d+LDM|0.935(.001)|0.940(.002)|0.939(.002)|0.927(.001)|0.898(.009)|0.954(.001)|0.932(.003)|
|LSTM_bidir|0.914(.003)|0.924(.004)|0.929(.002)|0.921(.001)|0.865(.008)|0.951(.001)|0.917(.003)|
|LSTM_bidir+LDM|0.932(.001)|0.936(.003)|0.936(.001)|0.926(.001)|0.876(.011)|0.953(.001)|0.927(.003)|
|Resnet1d_wang|0.919(.001)|0.925(.005)|0.929(.003)|0.926(.000)|0.881(.007)|0.941(.002)|0.920(.003)|
|Resnet1d_wang+LDM|0.930(.000)|0.942(.002)|0.941(.000)|0.932(.000)|0.889(.006)|0.945(.001)|0.930(.002)|
|FCN_wang|0.912(.001)|0.922(.000)|0.924(.002)|0.924(.000)|0.862(.009)|0.920(.006)|0.911(.003)|
|FCN_wang+LDM|0.917(.001)|0.930(.003)|0.936(.003)|0.928(.001)|0.875(.003)|0.928(.001)|0.919(.002)|
|XResNet1d101|0.924(.002)|0.933(.002)|0.926(.001)|0.928(.000)|0.895(.006)|0.955(.001)|0.927(.002)|
|XResNet1d101+LDM|0.937(.002)|0.939(.001)|0.935(.002)|0.932(.002)|0.906(.004)|0.957(.000)|0.934(.002)|

### Citation
If you think this repo is useful in your research, please consider citing our paper.
```
@article{ZHANG2023110298,
title = {Label decoupling strategy for 12-lead ECG classification},
journal = {Knowledge-Based Systems},
volume = {263},
pages = {110298},
year = {2023},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2023.110298},
url = {https://www.sciencedirect.com/science/article/pii/S0950705123000485},
author = {Shuo Zhang and Yuwen Li and Xingyao Wang and Hongxiang Gao and Jianqing Li and Chengyu Liu},
}
```
Meanwhile, our implemetation uses parts of some public codes in [Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL](https://ieeexplore.ieee.org/document/9190034). Please consider citing this paper.
```
@ARTICLE{9190034,
  author={Strodthoff, Nils and Wagner, Patrick and Schaeffter, Tobias and Samek, Wojciech},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL}, 
  year={2021},
  volume={25},
  number={5},
  pages={1519-1528},
  doi={10.1109/JBHI.2020.3022989}}
```

