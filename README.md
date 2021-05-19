## Efficient Person ReID system based on Active Learning 
Origin ReID-Model: [Learning Discriminative Features with Multiple Granularities for Person Re-Identification](https://arxiv.org/abs/1804.01438v1)

Active Learning: Uncertainty Sampling (Least Confidence)

## Dependencies

- Python >= 3.5
- PyTorch >= 0.4.0
- torchvision
- scipy
- numpy
- scikit_learn

## Data
The data structure would look like:
```
data/
    bounding_box_train/
    bounding_box_test/
    query/
```

#### Market1501 
Download from [here](http://www.liangzheng.org/Project/project_reid.html)

#### DukeMTMC-reID
Download from [here](http://vision.cs.duke.edu/DukeMTMC/)

#### CUHK03
1. Download cuhk03 dataset from "http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html"
2. Unzip the file and you will get the cuhk03_release dir include cuhk-03.mat
3. Download "cuhk03_new_protocol_config_detected.mat" from "https://github.com/zhunzhong07/person-re-ranking/tree/master/evaluation/data/CUHK03"
and put it with cuhk-03.mat. We need this new protocol to split the dataset.
```
python utils/transform_cuhk03.py --src <path/to/cuhk03_release> --dst <path/to/save>
```

NOTICE:You need to change num_classes in network depend on how many people in your train dataset! e.g. 751 in Market1501

## Weights
Pretrained weight download from [here](https://drive.google.com/open?id=16V7ZsflBbINHPjh_UVYGBVO6NuSxEMTi)

## Train
```
python main.py --mode train --data_path <path/to/Market-1501-v15.09.15> 
```

## Evaluate
```
python main.py --mode evaluate --data_path <path/to/Market-1501-v15.09.15> --weight <path/to/weight_name.pt> 
```

## Citation of Origin MGN
```text
@ARTICLE{2018arXiv180401438W,
    author = {{Wang}, G. and {Yuan}, Y. and {Chen}, X. and {Li}, J. and {Zhou}, X.},
    title = "{Learning Discriminative Features with Multiple Granularities for Person Re-Identification}",
    journal = {ArXiv e-prints},
    year = 2018,
}
```
