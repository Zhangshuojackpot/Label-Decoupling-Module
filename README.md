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
Run [get_ptbxl.sh]()
