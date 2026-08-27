import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

def gerar_dados_partidas(n_partidas=500):
    """Gera dados simulados de partidas para o nosso dataset."""
    np.random.seed(42)
    
    posse_de_bola = np.random.randint(35, 70, n_partidas) 
    chutes_no_gol = np.random.randint(2, 15, n_partidas)
    desarmes = np.random.randint(10, 35, n_partidas)
    fator_casa = np.random.choice([0, 1], n_partidas) # 1 = Allianz, 0 = Fora
    
    # Lógica de pontuação para definir vitória
    score_forca = (chutes_no_gol * 1.5) + (desarmes * 0.8) + (fator_casa * 5) - (posse_de_bola * 0.1)
    vitoria = np.where(score_forca > np.median(score_forca), 1, 0)
    
    df = pd.DataFrame({
        'posse_de_bola': posse_de_bola,
        'chutes_no_gol': chutes_no_gol,
        'desarmes': desarmes,
        'fator_casa': fator_casa,
        'vitoria': vitoria
    })
    return df

def main():
    print("Iniciando o processamento dos dados do Maior Campeão do Brasil... 🏆\n")
    
    # 1. Carregando a base
    df = gerar_dados_partidas()
    
    # 2. Separando Features (X) e Target (y)
    X = df.drop('vitoria', axis=1)
    y = df['vitoria']
    
    # 3. Divisão Treino/Teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 4. Padronização (Crucial para o SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Instanciando e treinando o SVM com Kernel Polinomial
    print("Treinando a máquina com Kernel Polinomial (degree=3)... 🧠")
    modelo_svm_poly = SVC(kernel='poly', degree=3, coef0=1.0, C=1.0, gamma='scale')
    modelo_svm_poly.fit(X_train_scaled, y_train)
    
    # 6. Avaliação do Modelo
    previsoes = modelo_svm_poly.predict(X_test_scaled)
    
    print("\n--- Resultados ---")
    print(f"Acurácia: {accuracy_score(y_test, previsoes) * 100:.2f}%\n")
    print("Relatório Completo:")
    print(classification_report(y_test, previsoes))

if __name__ == "__main__":
    main()