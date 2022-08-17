from torch.distributed import get_rank


def is_mainprocess():
    return get_rank() == 0
