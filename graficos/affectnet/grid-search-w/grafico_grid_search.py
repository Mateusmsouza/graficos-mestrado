import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

file_path = './grid_search_affectnet.csv'
df = pd.read_csv(file_path)

# Exemplo de dados
W = df['w']
acc = df['Accuracy/test (Max)']
prob = df['prob']
prob = prob.fillna(0.2)

# Ajusta o tamanho da bolha (raio ~ área)
sizes = [p * 2000 for p in prob]  # fator de escala para visualização
'''
plt.figure(figsize=(8, 6))
scatter = plt.scatter(W, acc, s=sizes, alpha=0.6, c=prob, cmap="viridis", edgecolors="k")

plt.xlabel("W (Parâmetro)")
plt.ylabel("Acurácia Máxima")
plt.title("Relação entre W, Acurácia e Probabilidade")
plt.colorbar(scatter, label="Probabilidade (escala de cores)")
plt.show()
'''
# Mapeamento manual de cor por categoria
colors = {0.2: "red", 0.4: "blue", 0.6: "green", 0.8: "yellow", 0.9: "black"}

plt.figure(figsize=(8, 6))
for p in set(prob):
    mask = [i for i, val in enumerate(prob) if val == p]
    plt.scatter(
        [W[i] for i in mask],
        [acc[i] for i in mask],
        s=[prob[i] * 2000 for i in mask],
        c=colors[p],
        alpha=0.7,
        edgecolors="k",
        label=f"Prob = {p}"
    )

plt.xlabel("W (Parâmetro)")
plt.ylabel("Acurácia Máxima")
plt.title("Relação entre W, Acurácia e Probabilidade")
plt.legend(title="Probabilidade", bbox_to_anchor=(1.05, 1),
           loc="upper left", markerscale=0.4, labelspacing=1.5)
plt.tight_layout()
plt.savefig("plot.png")
