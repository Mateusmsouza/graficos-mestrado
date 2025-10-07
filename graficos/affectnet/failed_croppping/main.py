import json
import sys
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

report = sys.argv[1]
print(f"opening file {report}")
with open(report, 'r') as file:
    data = json.load(file)

print(data.keys())

df_crop_report_number = pd.DataFrame(
    {"success_mix": data["faces_with_failed_components_crop_paste"]})

failed_mix = df_crop_report_number[df_crop_report_number["success_mix"] == 0]

print(f"from {len(df_crop_report_number)} images, {len(failed_mix)}"
      " images failed facemixup mixing due to bad components")

# Soma os valores de cada sublista
sums = [sum(sublist)
        for sublist in data["faces_with_failed_components_crop_paste_details"]]

# Ordena do menor para o maior
sorted_sums = sorted(sums)


# Calcula os percentis (CDF)

# Plota o gráfico
plt.figure(figsize=(6, 4))
plt.plot(y=sorted_sums, marker='o')
plt.title("Gráfico de Percentil (CDF)")
plt.xlabel("Soma por sublista")
plt.ylabel("Percentil (%)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("foo.png")
