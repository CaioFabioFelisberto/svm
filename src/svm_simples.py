import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Criando nossa base própria do zero (dados fictícios)
dados = {
    'posse_de_bola': [45, 60, 55, 30, 65, 40, 50, 70, 35, 58],
    'chutes_no_gol': [3, 8, 5, 2, 10, 4, 6, 9, 1, 7],
    'vitoria': [0, 1, 1, 0, 1, 0, 0, 1, 0, 1] # 1 = Ganhou, 0 = Não Ganhou
}

df = pd.DataFrame(dados)

# 2. Separando as features (X) e o alvo (y)
X = df[['posse_de_bola', 'chutes_no_gol']]
y = df['vitoria']

# 3. Dividindo em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Instanciando o monstro (usando Kernel Linear pra começar simples)
modelo_svm = SVC(kernel='linear')

# 5. Treinando a máquina
modelo_svm.fit(X_train, y_train)

# 6. Prevendo e vendo se o modelo é brabo
previsoes = modelo_svm.predict(X_test)
print(f"Acurácia do modelo: {accuracy_score(y_test, previsoes) * 100}%")