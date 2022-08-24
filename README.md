# PointNet2
This repository is implementation of [PointNet++](https://arxiv.org/abs/1706.02413) with PyTorch.

### Supporting variants
- PointNet++ classification with SSG
- PointNet++ classification with MSG

### Data preparation
#### ModelNet
For ModelNet classification, you can download dataset [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and save in ```datasets/modelnet```.

#### ShapeNet
For ShapeNet part segmentation, you can download dataset [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip) and save in ```datasets/shapenet```.

#### Directory
```
datasets/
    modelnet/
    shapenet/
```

### Training
```console
python train.py --model pointnet2_cls_ssg
```
For the ```--model``` option, you can give one of models in [modeling/architectures](https://github.com/hanchaa/PointNet2/tree/main/modeling/architectures).

Options for running scripts can be found with ```--help``` option.

### Evaluation
```console
python train.py --model pointnet2_cls_ssg --eval-only
```

### Reference
[yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)