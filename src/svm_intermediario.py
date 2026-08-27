import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Semente pra garantir que o bagulho seja reproduzível
np.random.seed(42)

# Gerando 500 partidas simuladas do Verdão
n_partidas = 500

# Criando as features (X)
posse_de_bola = np.random.randint(35, 70, n_partidas) 
chutes_no_gol = np.random.randint(2, 15, n_partidas)
desarmes = np.random.randint(10, 35, n_partidas)
fator_casa = np.random.choice([0, 1], n_partidas) # 1 = Allianz Parque, 0 = Fora

# Uma lógica inventada pra definir a variável alvo (y)
# Mais chutes, mais desarmes e jogar em casa aumentam o "score" de vitória
score_forca = (chutes_no_gol * 1.5) + (desarmes * 0.8) + (fator_casa * 5) - (posse_de_bola * 0.1)
vitoria = np.where(score_forca > np.median(score_forca), 1, 0) # 1 = Ganhou, 0 = Não ganhou

# Montando o DataFrame
df_palmeiras = pd.DataFrame({
    'posse_de_bola': posse_de_bola,
    'chutes_no_gol': chutes_no_gol,
    'desarmes': desarmes,
    'fator_casa': fator_casa,
    'vitoria': vitoria
})

# Separando X e y
X = df_palmeiras.drop('vitoria', axis=1)
y = df_palmeiras['vitoria']

# Dividindo em treino e teste (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# PADRONIZANDO OS DADOS (O pulo do gato pro SVM!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Treinando o monstro com Kernel RBF (Radial Basis Function - pra pegar os padrões não-lineares)
modelo_svm = SVC(kernel='rbf', C=1.0, gamma='scale')
modelo_svm.fit(X_train_scaled, y_train)

# Prevendo e avaliando
previsoes = modelo_svm.predict(X_test_scaled)
print(f"Acurácia: {accuracy_score(y_test, previsoes) * 100:.2f}%\n")
print("Relatório de Classificação Completo:")
print(classification_report(y_test, previsoes))