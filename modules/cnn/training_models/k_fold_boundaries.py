"""Simple function to return the boundaries for the different folds"""


def k_fold_boundaries(values, folds):
    """Take a list of values and number of folds, return equally spaced boundaries as tuples"""
    return [
        (int((i / folds) * len(values)), int(((i + 1) / folds) * (len(values))))
        for i in range(folds)
    ]


if __name__ == "__main__":

    l1 = list(range(50))

    fold_boundaries = k_fold_boundaries(l1, 5)

    v1 = l1[fold_boundaries[0][0] : fold_boundaries[0][1]]
    print(v1)
    v2 = l1[fold_boundaries[1][0] : fold_boundaries[1][1]]
    print(v2)
    v3 = l1[fold_boundaries[2][0] : fold_boundaries[2][1]]
    print(v3)
    v4 = l1[fold_boundaries[3][0] : fold_boundaries[3][1]]
    print(v4)
    v5 = l1[fold_boundaries[4][0] : fold_boundaries[4][1]]
    print(v5)

    t1 = l1[: fold_boundaries[0][0]] + l1[fold_boundaries[0][1] :]
    print(t1)
    t2 = l1[: fold_boundaries[1][0]] + l1[fold_boundaries[1][1] :]
    print(t2)
    t3 = l1[: fold_boundaries[2][0]] + l1[fold_boundaries[2][1] :]
    print(t3)
    t4 = l1[: fold_boundaries[3][0]] + l1[fold_boundaries[3][1] :]
    print(t4)
    t5 = l1[: fold_boundaries[4][0]] + l1[fold_boundaries[4][1] :]
    print(t5)
