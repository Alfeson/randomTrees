import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class DataLoader:
    def __init__(self, train_path, test_path):
        self.train = pd.read_csv(train_path)
        self.test = pd.read_csv(test_path)


class Preprocessor:
    def __init__(self, train, test):
        self.train = train
        self.test = test
        self.num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        self.cat_cols = ['CryoSleep', 'VIP']
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    def fill_missing_values(self):
        for col in self.num_cols:
            median = self.train[col].median()
            self.train[col] = self.train[col].infer_objects(median)
            self.test[col] = self.test[col].infer_objects(median)

        for col in self.cat_cols:
            mode = self.train[col].mode()[0]
            self.train[col] = self.train[col].infer_objects(mode).astype(str)
            self.test[col] = self.test[col].infer_objects(mode).astype(str)

        self.train['HomePlanet'] = self.train['HomePlanet'].infer_objects('Earth')
        self.test['HomePlanet'] = self.test['HomePlanet'].infer_objects('Earth')

        self.train['Destination'] = self.train['Destination'].infer_objects('TRAPPIST-1e')
        self.test['Destination'] = self.test['Destination'].infer_objects('TRAPPIST-1e')

        self.train['Cabin'] = self.train['Cabin'].infer_objects('G/0/S')
        self.test['Cabin'] = self.test['Cabin'].infer_objects('G/0/S')

    def encode_features(self):
        cat_features = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']

        self.train[cat_features] = self.train[cat_features].astype(str)
        self.test[cat_features] = self.test[cat_features].astype(str)

        self.ohe.fit(self.train[cat_features])

        train_cat = pd.DataFrame(
            self.ohe.transform(self.train[cat_features]),
            columns=self.ohe.get_feature_names_out(cat_features)
        )
        test_cat = pd.DataFrame(
            self.ohe.transform(self.test[cat_features]),
            columns=self.ohe.get_feature_names_out(cat_features)
        )

        train_cat.reset_index(drop=True, inplace=True)
        test_cat.reset_index(drop=True, inplace=True)
        self.train.reset_index(drop=True, inplace=True)
        self.test.reset_index(drop=True, inplace=True)

        train_final = pd.concat([self.train[self.num_cols], train_cat], axis=1)
        test_final = pd.concat([self.test[self.num_cols], test_cat], axis=1)

        return train_final, test_final


class Model:
    def __init__(self):
        self.clf = RandomForestClassifier()

    def train(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)


class Submission:
    def __init__(self, test_df, predictions, filename='submission.csv'):
        self.test_df = test_df
        self.predictions = predictions
        self.filename = filename

    def save(self):
        submission_df = pd.DataFrame({
            'PassengerId': self.test_df['PassengerId'],
            'Transported': self.predictions
        })
        submission_df.to_csv(self.filename, index=False)


# Pipeline
if __name__ == '__main__':
    # Carrega os dados
    data = DataLoader("../dataset/train.csv", "../dataset/test.csv")

    # Divide em treino/validação para calcular acurácia
    train_df, val_df = train_test_split(data.train, test_size=0.2, random_state=42)

    # Pré-processamento
    prep = Preprocessor(train_df.copy(), val_df.copy())
    prep.fill_missing_values()
    X_train, X_val = prep.encode_features()

    y_train = train_df['Transported']
    y_val = val_df['Transported']

    # Modelo
    model = Model()
    model.train(X_train, y_train)
    val_predictions = model.predict(X_val)

    # Cálculo da acurácia
    acc = accuracy_score(y_val, val_predictions)
    print(f"Acurácia na validação: {acc:.4f}")
