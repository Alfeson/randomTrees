# Importação de bibliotecas essenciais
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Implementação de uma Random Forest personalizada
class RandomForest:
    def __init__(self, n_trees=100, max_depth=None, max_features='sqrt', random_state=None):
        self.n_trees = n_trees  # Número de árvores na floresta
        self.max_depth = max_depth  # Profundidade máxima de cada árvore
        self.max_features = max_features  # Máximo de features a serem usadas por árvore
        self.random_state = random_state  # Semente para reprodutibilidade
        self.trees = []  # Lista que armazena as árvores e os índices das features usadas

    def fit(self, X, y):
        # Conversão para numpy array se necessário
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
        y = np.array(y)

        self.trees = []
        n_samples, n_features = X.shape

        # Determina o número de features a usar por árvore
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        elif isinstance(self.max_features, int):
            max_features = self.max_features
        else:
            max_features = n_features

        rng = np.random.RandomState(self.random_state)

        for _ in range(self.n_trees):
            # Bootstrap: amostragem com reposição de exemplos
            indices = rng.choice(n_samples, n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            # Seleciona um subconjunto aleatório de features
            feature_indices = rng.choice(n_features, max_features, replace=False)

            # Cria e treina uma árvore com os dados amostrados
            tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(X_sample[:, feature_indices], y_sample)

            # Armazena a árvore treinada e os índices das features utilizadas
            self.trees.append((tree, feature_indices))

    def predict(self, X):
        # Conversão para numpy array se necessário
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()

        # Inicializa matriz para armazenar as previsões de cada árvore
        predictions = np.zeros((X.shape[0], len(self.trees)), dtype=int)

        # Cada árvore faz a predição em seu subconjunto de features
        for i, (tree, feature_indices) in enumerate(self.trees):
            predictions[:, i] = tree.predict(X[:, feature_indices])

        # Retorna a classe mais votada entre as árvores (votação majoritária)
        final_predictions = np.apply_along_axis(lambda row: np.bincount(row).argmax(), axis=1, arr=predictions)
        return final_predictions


# ---------------------------
# Exemplo de uso real com os dados da competição Spaceship Titanic
# ---------------------------
if __name__ == "__main__":
    # Carrega os arquivos CSV
    train = pd.read_csv("../dataset/train.csv")
    test = pd.read_csv("../dataset/test.csv")

    # Preenche valores nulos com o último valor válido (forward fill)
    train.fillna(method="ffill", inplace=True)
    test.fillna(method="ffill", inplace=True)

    # Colunas categóricas que precisam ser convertidas para numéricas
    categorical_cols = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']
    for col in categorical_cols:
        # Codifica os dados combinando treino e teste (para evitar erros de valores novos)
        all_values = pd.concat([train[col], test[col]]).astype(str)
        le = LabelEncoder()
        le.fit(all_values)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

    # Define as colunas que serão usadas como entrada do modelo
    features = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age',
                'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    X = train[features]
    y = train["Transported"].astype(bool)  # Alvo binário
    X_test = test[features]

    # Divide os dados em treino e validação para avaliar desempenho
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instancia e treina o modelo RandomForest personalizado
    rf = RandomForest(n_trees=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)

    # Avalia o modelo na base de validação
    y_pred = rf.predict(X_val)
    print("Acurácia na validação:", accuracy_score(y_val, y_pred))

    # Gera predições no conjunto de teste
    test_preds = rf.predict(X_test)

    # Cria e salva o arquivo de submissão
    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Transported": test_preds
    })
    submission.to_csv("submission_random_forest_custom.csv", index=False)
    print("Arquivo 'submission_random_forest_custom.csv' gerado com sucesso.")
