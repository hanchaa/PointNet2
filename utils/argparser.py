import argparse


def parse_args():
    parser = argparse.ArgumentParser("Training PointNet++")
    parser.add_argument("--num-gpus", type=int, help="the number of gpus for training")
    parser.add_argument("--model", type=str, help="name of model to train")
    parser.add_argument("--dataset", type=str, default="", help="dataset for training and evaluation")
    parser.add_argument("--num-point", type=int, default=1024, help="the number of points to sample per shape")
    parser.add_argument("--use-normal", type=bool, default=False, help="whether to use normal for point features")
    parser.add_argument("--epoch", type=int, default=100, help="the number of epochs in training")
    parser.add_argument("--batch-size", type=int, default=16, help="the number of total batch in training")
    parser.add_argument("--test-batch-size", type=int, default=1, help="the number of total batch in testing")
    parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="strength of weight decay regularization")
    parser.add_argument("--checkpoint-period", type=int, default=5, help="period of saving checkpoint of the training")
    parser.add_argument("--model-weights", type=str, help="weight file of the model")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="path for logging")
    parser.add_argument("--resume", default=False, action="store_true", help="resume training from checkpoint")
    parser.add_argument("--eval-only", default=False, action="store_true", help="evaluate model only")
    parser.add_argument("--wandb", type=str, default="", help="id of wandb. leave blank if you don't use wandb")
    parser.add_argument("--wandb-run-id", type=str, default="", help="run id of wandb to resume logging")
    return parser.parse_args()
