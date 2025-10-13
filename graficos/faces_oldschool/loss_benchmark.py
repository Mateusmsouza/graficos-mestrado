import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from cycler import cycler


def _create_exp_dict(name, file):
    return {
        "exp_name": name,
        "df": pd.read_csv(file)
    }


all_dfs = [
    _create_exp_dict(
        "FaceMixup", "./runs_loss/efficientNetb0_classify_by_faces_MIXED_2_fold-1_Loss_train.csv"),
    _create_exp_dict(
        "MixAugment", "./runs_loss/efficientNetb0_classify_by_faces_MIXUPAUGMENT_0T_1_fold-1_Loss_train.csv"),
    _create_exp_dict(
        "CutMix", "./runs_loss/efficientNetb0_classify_by_faces_CUTMIX_0T_1-REMAKE_fold-1_Loss_train.csv"),
    _create_exp_dict(
        "Random Erasing", "./runs_loss/efficientNetb0_classify_by_faces_BASELINE_RANDOM_ERASING_1_fold-1_Loss_train.csv"),
    _create_exp_dict(
        "MixUp", "./runs_loss/efficientNetb0_classify_by_faces_MIXUP_1_fold-1_Loss_train.csv"),
    _create_exp_dict(
        "Vanilla", "./runs_loss/baseline_0t.csv"),
    _create_exp_dict(
        "Vanilla with Mixed Faces", "./runs_loss/efficientNetb0_classify_by_faces_BASELINE_6T_1_fold-1_Loss_train.csv"),
]
