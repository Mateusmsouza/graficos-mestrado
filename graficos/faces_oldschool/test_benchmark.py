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
all_dfs = [
    {"exp_name": "FaceMixup", "df": face_mix},
    {"exp_name": "MixAugment", "df": mixedupaugment},
    {"exp_name": "Vanilla Auto Augment", "df": baseline_0t_da},
    {"exp_name": "CutMix", "df": cutmix},
    {"exp_name": "Vanilla Cutout", "df": baselina_0t_cutout},
    {"exp_name": "Random Erasing", "df": baseline_random_erasing},
    {"exp_name": "MixUp", "df": mixup},
    {"exp_name": "Vanilla", "df": baseline_0t},
    {"exp_name": "Vanilla Mixed + Auto Augment", "df": baseline_6t_da},
    {"exp_name": "Vanilla Mixed + Cutout", "df": baselina_6t_cutout},
    {"exp_name": "MixUp Cutout", "df": mixup_cutout},
    {"exp_name": "Vanilla Mixed", "df": baseline_6t},
    {"exp_name": "Mixup Auto Augment", "df": mixup_da},
]


def print_max_accuracy(df, approach_name):
    print(f"[{approach_name}] Best acc is {df['Value'].max()}"
          f" in Epoch {df['Value'].idxmax()}")


for experiment in all_dfs:
    print_max_accuracy(experiment['df'], experiment['exp_name'])


# plot graph

df = {
    # "Step": pd.read_csv(curves[0])["Step"],
    "FaceMixup": face_mix["Value"],
    "MixAugment": mixedupaugment["Value"],
    "CutMix": cutmix["Value"],
    "Random Erasing": baseline_random_erasing["Value"],
    "MixUp": mixup["Value"],
    "Vanilla": baseline_0t["Value"],
    "Vanilla with Mixed Faces": baseline_6t["Value"]
}
df = pd.DataFrame(df)
sns.set_theme(style='white', font_scale=1)
sns.color_palette("rocket_r")

plt.figure(figsize=(8, 6))
facemixup_color = r"#3357FF"  # 4990c2"
facemixup_rs_color = r"#ff8e2b"
mixaugment_color = r"#f02041"
cutmix_color = r"#F39C12"
random_erasing_color = r"#8E44AD"
mixup = r"#2ECC71"
vanilla_6k = r"#9d402f"
vanilla = "#7a9e2f"
custom_palette = [facemixup_color, mixaugment_color,
                  cutmix_color, random_erasing_color, mixup, vanilla_6k, vanilla]

ax = sns.lineplot(data=df, sort=True, palette=custom_palette)
ax.set(xlabel='Epoch', ylabel='Accuracy')


plt.title("Approaches")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig("approaches.png")
