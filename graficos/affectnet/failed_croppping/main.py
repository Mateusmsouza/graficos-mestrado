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
# Agrupa e conta quantas vezes cada soma apareceu
value_counts = pd.Series(sums).value_counts().sort_index()

# Exibe a tabela (opcional)
print(value_counts)

# Plota o gráfico de barras
plt.figure(figsize=(7, 5))
value_counts.plot(kind='bar', color='steelblue', edgecolor='black')
plt.title("Distribuição de Somas (Contagem por Valor)")
plt.xlabel("Valor da soma por sublista")
plt.ylabel("Quantidade de ocorrências")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("foo_counts.png")
plt.show()