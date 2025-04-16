import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score

class RandomForest:
    def __init__(self, n_trees=100, max_depth=None, max_features='sqrt'):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []
    
    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        # Determinar o número de features para cada árvore
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        else:
            max_features = n_features
        
        for _ in range(self.n_trees):
            # Amostragem bootstrap
            X_sample, y_sample = resample(X, y)
            
            # Selecionar subconjunto de features
            feature_indices = np.random.choice(n_features, max_features, replace=False)
            
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X_sample[:, feature_indices], y_sample)
            
            self.trees.append((tree, feature_indices))
    
    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.trees)))
        
        for i, (tree, feature_indices) in enumerate(self.trees):
            predictions[:, i] = tree.predict(X[:, feature_indices])
        
        # Votação majoritária
        return np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=predictions)