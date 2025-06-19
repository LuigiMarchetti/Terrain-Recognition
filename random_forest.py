import numpy as np
import os
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt

# Defina os diretórios
DATASET_DIR = "./EuroSAT_RGB"
CLASSES_VERDES = ['Forest', 'Pasture', 'HerbaceousVegetation']  # classes que representam áreas verdes

# Função para carregar imagens RGB e rótulos
def load_data():
    X, y = [], []
    for class_name in os.listdir(DATASET_DIR):
        class_path = os.path.join(DATASET_DIR, class_name)
        if not os.path.isdir(class_path):
            continue
        for file in os.listdir(class_path):
            if file.endswith(".jpg") or file.endswith(".png"):
                img_path = os.path.join(class_path, file)
                with rasterio.open(img_path) as img:
                    data = img.read([1, 2, 3])  # R, G, B
                    data = data.reshape(3, -1).mean(axis=1)  # média por banda
                    X.append(data)
                    y.append(1 if class_name in CLASSES_VERDES else 0)  # 1 para verde, 0 para não-verde
    return np.array(X), np.array(y)

# Carrega os dados
X, y = load_data()

# Divide em treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializa RandomForest com warm_start para acompanhar progresso
clf = RandomForestClassifier(
    n_estimators=1,
    warm_start=True,
    max_depth=10,
    random_state=42
)

total_estimators = 100
accuracies = []

print("Treinando Random Forest:")
for i in tqdm(range(1, total_estimators + 1)):
    clf.set_params(n_estimators=i)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

# Avaliação final
print("\nRelatório final:")
print(classification_report(y_test, clf.predict(X_test)))

# Salva o modelo
os.makedirs("model", exist_ok=True)
joblib.dump(clf, "model/random_forest_areas_verdes.pkl")

# Mostra gráfico da evolução da acurácia
plt.plot(range(1, total_estimators + 1), accuracies, marker='o')
plt.xlabel("Número de Árvores")
plt.ylabel("Acurácia no conjunto de teste")
plt.title("Evolução da Acurácia durante o Treinamento")
plt.grid(True)
plt.tight_layout()
plt.show()
