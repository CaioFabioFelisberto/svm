# Classificação de partidas com SVM

Projeto didático que demonstra como usar **Support Vector Machines (SVM)** para classificar o resultado de partidas de futebol. Os dados são gerados no próprio código e representam um cenário fictício baseado em estatísticas como posse de bola, chutes no gol, desarmes e mando de campo.

## Objetivo

Apresentar uma evolução gradual do uso de um classificador `SVC` do scikit-learn:

- **Simples:** pequeno conjunto de dados criado manualmente e kernel linear.
- **Intermediário:** 500 partidas simuladas, padronização das variáveis e kernel RBF.
- **Complexo:** mesma ideia do exemplo intermediário, reorganizada em funções e usando kernel polinomial.

> Os dados e a regra que define a vitória são artificiais. Os resultados servem apenas para estudo e não representam previsões reais de partidas.

## Tecnologias

- Python 3
- NumPy
- pandas
- scikit-learn

## Como executar

1. Clone ou baixe este repositório e abra a pasta do projeto no terminal.
2. Crie e ative um ambiente virtual:

   ```bash
   python -m venv .venv
   ```

   No Windows PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   No Linux/macOS:

   ```bash
   source .venv/bin/activate
   ```

3. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

4. Execute um dos exemplos:

   ```bash
   python src/svm_simples.py
   python src/svm_intermediario.py
   python src/svm_complexo.py
   ```

Cada script exibe a acurácia do modelo. As versões intermediária e complexa também exibem o relatório de classificação, com precisão, revocação e F1-score.

## Como os dados são gerados

As versões intermediária e complexa criam 500 partidas com valores aleatórios para:

| Variável | Descrição |
| --- | --- |
| `posse_de_bola` | Percentual simulado de posse de bola |
| `chutes_no_gol` | Quantidade de chutes no gol |
| `desarmes` | Quantidade de desarmes |
| `fator_casa` | `1` para jogar em casa e `0` fora |
| `vitoria` | Alvo: `1` para vitória e `0` para não vitória |

O alvo `vitoria` é calculado a partir de uma pontuação inventada. Depois, os dados são divididos em treino e teste, padronizados com `StandardScaler` e usados no treinamento do SVM.

## Estrutura

```text
.
├── requirements.txt
├── README.md
└── src/
    ├── svm_simples.py
    ├── svm_intermediario.py
    └── svm_complexo.py
```

## Próximos passos

- Substituir os dados simulados por partidas reais.
- Comparar diferentes kernels e valores de `C` e `gamma`.
- Usar validação cruzada e outras métricas além da acurácia.
- Salvar o modelo treinado para realizar novas classificações.