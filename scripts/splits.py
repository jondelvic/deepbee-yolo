import random
import math

# splits an annotation dict into train and val subsets at image level rather than tile level to prevent data leakage
# note: overlapping tiles from the same source image cannot appear in both train and val

# returns (train_dict, val_dict)

def split_train_val(
    annotations: dict,
    val_fraction: float,
    seed: int,
) -> tuple[dict, dict]:
    rng   = random.Random(seed)
    names = list(annotations.keys())
    rng.shuffle(names)
    n_val = max(1, math.ceil(len(names) * val_fraction))
    val   = {k: annotations[k] for k in names[:n_val]}
    train = {k: annotations[k] for k in names[n_val:]}
    return train, val