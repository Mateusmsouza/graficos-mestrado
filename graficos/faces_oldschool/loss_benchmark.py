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

df = {
    exp["exp_name"]: exp["df"]["Value"]
    for exp in all_dfs
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
ax.set(xlabel='Epoch', ylabel='Loss')


plt.title("Approaches")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig("approaches_losses.png")
