# PointNet2
This repository is implementation of [PointNet++](https://arxiv.org/abs/1706.02413)

### Supporting variants
- PointNet++ classification with SSG
- PointNet++ classification with MSG

### Training
```console
python train.py --model pointnet2_cls_ssg
```
For the ```--model``` option, you can give one of models in [modeling/architectures](https://github.com/hanchaa/PointNet2/tree/main/modeling/architectures)

Options for running scripts can be found with ```--help``` option

### Evaluation
```console
python train.py --model pointnet2_cls_ssg --eval-only
```
