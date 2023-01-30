# Label-Decoupling-Module
This is the official PyTorch implementation of our work [Label decoupling strategy for 12-lead ECG classification](https://www.sciencedirect.com/science/article/pii/S0950705123000485), which has published in Knowledge-Based Systems. This repo contains some key codes of our LDM and its application in PTB-XL dataset.<br>
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
cd ./code_used_upload
python order_results_ptbxl1_upload.py
```

### Results
|Method|All|Diag.|Sub-diag.|Super-diag.|Form|Rhythm
|:---|:---|:---|:---|:---|:---|:---|
|LSTM|0.909(.002)|0.926(.002)|0.926(.002)||||
|LSTM+LDM|0.923(.001)|0.931(.002)|0.935(.003)||||
|Inception1d|0.925(.002)|0.928(.000)|0.927(.000)|||
|Inception1d+LDM|0.935(.001)|0.940(.002)|0.939(.002)|||
|LSTM_bidir|0.914(.003)|0.924(.004)|0.929(.002)|||
|LSTM_bidir+LDM|0.932(.001)|0.936(.003)|0.936(.001)|||
|Resnet1d_wang|0.919(.001)|0.925(.005)|0.929(.003)|||
|Resnet1d_wang+LDM|0.930(.000)|0.942(.002)|0.941(.000)|||
|FCN_wang|0.912(.001)|0.922(.000)|0.924(.002)|||
|FCN_wang+LDM|0.917(.001)|0.930(.003)|0.936(.003)|||
|XResNet1d101|0.924(.002)|0.933(.002)|0.926(.001)|||
|XResNet1d101+LDM|0.937(.002)|0.939(.001)|0.935(.002)|||
