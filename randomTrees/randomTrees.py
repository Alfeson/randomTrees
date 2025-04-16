import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


class DataLoader:
    def __init__(self, train_path, test_path):
        self.train = pd.read_csv(train_path)
        self.test = pd.read_csv(test_path)


class Preprocessor:
    def __init__(self):
        self.numerical_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        self.categorical_features = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']

    def preprocess_train(self, df):
        df = df.copy()
        # Separando Cabin
        df[['Cabin_Deck', 'Cabin_Num', 'Cabin_Side']] = df['Cabin'].str.split('/', expand=True)
        self.categorical_features.remove('Cabin')
        self.categorical_features.extend(['Cabin_Deck', 'Cabin_Side'])

        return df

    def preprocess_test(self, df):
        df = df.copy()
        # Separando Cabin
        df[['Cabin_Deck', 'Cabin_Num', 'Cabin_Side']] = df['Cabin'].str.split('/', expand=True)
        return df


def create_pipeline():
    numerical_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Cabin_Deck', 'Cabin_Side']

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'  # Manter outras colunas (PassengerId)
    )

    model = RandomForestClassifier(random_state=42)

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    return pipeline


class Submission:
    def __init__(self, test_df, predictions, filename='result.csv'):
        self.test_df = test_df
        self.predictions = predictions
        self.filename = filename

    def save(self):
        submission_df = pd.DataFrame({
            'PassengerId': self.test_df['PassengerId'],
            'Transported': self.predictions.astype(bool)
        })
        submission_df.to_csv(self.filename, index=False)
        print(f"Submission file saved to {self.filename}")


if __name__ == '__main__':
    # Carrega os dados
    data = DataLoader("../dataset/train.csv", "../dataset/test.csv")

    # Pré-processamento inicial para separar 'Cabin' antes da divisão
    preprocessor_initial = Preprocessor()
    train_df_processed = preprocessor_initial.preprocess_train(data.train.copy())
    test_df_processed = preprocessor_initial.preprocess_test(data.test.copy())

    # Divide em treino/validação
    train_df, val_df = train_test_split(train_df_processed, test_size=0.2, random_state=42,
                                        stratify=train_df_processed['Transported'])

    # Separa as features e a variável alvo
    X_train = train_df.drop('Transported', axis=1)
    y_train = train_df['Transported']
    X_val = val_df.drop('Transported', axis=1)
    y_val = val_df['Transported']
    X_test = test_df_processed.copy()

    # Cria o pipeline
    pipeline = create_pipeline()

    # Define os hiperparâmetros para busca
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 3, 5],
        'classifier__max_features': ['sqrt', 'log2'],
        'preprocessor__num__imputer__strategy': ['mean', 'median'],
        'preprocessor__cat__imputer__strategy': ['most_frequent', 'constant'],
        'preprocessor__cat__imputer__fill_value': ['Missing']  # Para 'constant'
    }

    # Realiza a busca por hiperparâmetros usando validação cruzada
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    # Melhores parâmetros encontrados
    print(f"Melhores hiperparâmetros encontrados: {grid_search.best_params_}")

    # Avalia o modelo com os melhores parâmetros no conjunto de validação
    best_model = grid_search.best_estimator_
    val_predictions = best_model.predict(X_val)
    accuracy = accuracy_score(y_val, val_predictions)
    print(f"Acurácia na validação com os melhores parâmetros: {accuracy:.4f}")

    # Treina o modelo final com todos os dados de treino e os melhores parâmetros
    final_model = RandomForestClassifier(**grid_search.best_params_['classifier'], random_state=42)

    # Refaz o pré-processamento no dataset de treino completo
    full_train_processed = preprocessor_initial.preprocess_train(data.train.copy())
    X_full_train = full_train_processed.drop('Transported', axis=1)
    y_full_train = full_train_processed['Transported']

    # Cria um pipeline de pré-processamento final
    final_preprocessor = create_pipeline().named_steps['preprocessor']
    X_full_train_transformed = final_preprocessor.fit_transform(X_full_train)
    X_test_transformed = final_preprocessor.transform(X_test)

    final_model.fit(X_full_train_transformed, y_full_train)

    # Faz as previsões no conjunto de teste
    test_predictions = final_model.predict(X_test_transformed)

    # Cria e salva o arquivo de submissão
    submission = Submission(data.test, test_predictions)
    submission.save()
