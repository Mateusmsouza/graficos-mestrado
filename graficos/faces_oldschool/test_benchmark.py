import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from cycler import cycler

curves = [
    'mixup/run-with-data-augmentation_runs_efficientNetb0_classify_by_faces_MIXUP_1_DA_fold-1_Accuracy_test-tag-fold-1.csv',
    'facemixup_offline/run-with-data-augmentation_runs_efficientNetb0_classify_by_faces_MIXED_1_DA_fold-1_Accuracy_test-tag-fold-1.csv',
    'vanilla/run-with-data-augmentation_runs_efficientNetb0_classify_by_faces_BASELINE_0T_1_DA_fold-1_Accuracy_test-tag-fold-1.csv',
    'mixup/run-without-data-augmentation_runs_efficientNetb0_classify_by_faces_MIXUP_1_fold-1_Accuracy_test-tag-fold-1.csv',
    'vanilla_mixed/run-with-data-augmentation_runs_efficientNetb0_classify_by_faces_BASELINE_6T_1_DA_fold-1_Accuracy_test-tag-fold-1.csv',
    'vanilla_mixed/run-without-data-augmentation_runs_efficientNetb0_classify_by_faces_BASELINE_6T_1_fold-1_Accuracy_test-tag-fold-1.csv',
    'facemixup_offline/run-without-data-augmentation_runs_efficientNetb0_classify_by_faces_MIXED_1_fold-1_Accuracy_test-tag-fold-1.csv',
    'vanilla/run-without-data-augmentation_runs_efficientNetb0_classify_by_faces_BASELINE_OT_1_fold-1_Accuracy_test-tag-fold-1.csv',
    'random_erasing/run-efficientNetb0_classify_by_faces_BASELINE_RANDOM_ERASING_1_fold-1_Accuracy_test-tag-fold-1.csv',
    'facemixup_offline/run-without-data-augmentation_runs_efficientNetb0_classify_by_faces_MIXED_2_fold-1_Accuracy_test-tag-fold-1.csv',
    'cutmix/run-without-data-augmentation_runs_efficientNetb0_classify_by_faces_CUTMIX_0T_1_fold-1_Accuracy_test-tag-fold-1.csv',
    'mixup/run-efficientNetb0_classify_by_faces_MIXUP_1_CUTOUT_fold-1_Accuracy_test-tag-fold-1.csv',
    'facemixup_offline/run-runs_efficientNetb0_classify_by_faces_MIXED_1_CUTOUT_fold-1_Accuracy_test-tag-fold-1.csv',
    'vanilla_mixed/run-runs_efficientNetb0_classify_by_faces_BASELINE_6T_1_CUTOUT_fold-1_Accuracy_test-tag-fold-1.csv',
    'vanilla/run-runs_efficientNetb0_classify_by_faces_BASELINE_0T_1_CUTOUT_fold-1_Accuracy_test-tag-fold-1.csv',
    'mixup/run-efficientNetb0_classify_by_faces_MIXUPAUGMENT_0T_1_fold-1_Accuracy_test-tag-fold-1.csv'
]


def get_path(run) -> str:
    return f"runs/{run}"


def get_file(curv_index):
    return pd.read_csv(get_path(curves[curv_index]))


mixup = get_file(3)
mixup_da = get_file(0)
baseline_0t_da = get_file(2)
baseline_6t_da = get_file(4)
baseline_0t = get_file(7)
baseline_6t = get_file(5)
baseline_random_erasing = get_file(8)
face_mix = get_file(9)
cutmix = get_file(10)
mixup_cutout = get_file(11)
mixed_cutout = get_file(12)
baselina_6t_cutout = get_file(13)
baselina_0t_cutout = get_file(14)
mixedupaugment = get_file(15)
