import numpy as np

class KNN:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
        self.count = 0

    def fit(self, X, y):
        self.X_train = np.array(X)  # Convert to numpy array for faster operations
        self.y_train = np.array(y)

    def euclidean_distance(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.X_train = np.asarray(self.X_train, dtype=np.float64)
        
        return np.sqrt(np.sum((X - self.X_train) ** 2, axis=1))
        # Efficient calculation of Euclidean distances for all points in X
        # return np.sqrt(np.sum((self.X_train - X) ** 2, axis=1))

    def manhattan_distance(self, X):
        # Efficient calculation of Manhattan distances for all points in X
        # return np.sum(np.abs(self.X_train - X), axis=1)
        X = np.asarray(X, dtype=np.float64)
        self.X_train = np.asarray(self.X_train, dtype=np.float64)
        return np.sum(np.abs(X - self.X_train), axis=1)

    def cosine_distance(self, X):
        dot_product = np.dot(self.X_train, X)
        norm_X_train = np.linalg.norm(self.X_train, axis=1)
        norm_X = np.linalg.norm(X)
        cosine_similarity = dot_product / (norm_X * norm_X_train)
        return 1 - cosine_similarity


        # Efficient calculation of Cosine distances for all points in X
        # dot_product = np.dot(self.X_train, X)
        # norm_X_train = np.linalg.norm(self.X_train, axis=1)
        # norm_X = np.linalg.norm(X)
        # return 1 - (dot_product / (norm_X_train * norm_X))

    def compute_distance(self, X):
        if self.distance_metric == 'euclidean':
            return self.euclidean_distance(X)
        elif self.distance_metric == 'manhattan':
            return self.manhattan_distance(X)
        elif self.distance_metric == 'cosine':
            return self.cosine_distance(X)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def predict(self, X):
        X = np.array(X)
        y_pred = np.zeros(X.shape[0], dtype=self.y_train.dtype)
        for i, x in enumerate(X):
            self.count += 1
            print("Prediction number ", self.count)
            distances = self.compute_distance(x)
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            labels, counts = np.unique(k_nearest_labels, return_counts=True)
            y_pred[i] = labels[np.argmax(counts)]
        
        return y_pred

    def accuracy(self, X_test, y_test):
        predictions = self.predict(X_test)
        self.count = 0
        return np.mean(predictions == y_test)

# Example usage:
# knn = KNN(k=5, distance_metric='euclidean')
# knn.fit(X_train, y_train)
# accuracy = knn.accuracy(X_test, y_test)

class Metrics:
    def __init__(self):
        pass

    @staticmethod
    def accuracy(y_true, y_pred):
        return np.mean(y_true == y_pred)

    @staticmethod
    def precision(y_true, y_pred, average='macro'):
        unique_labels = np.unique(y_true)
        precisions = []

        for label in unique_labels:
            true_positive = np.sum((y_true == label) & (y_pred == label))
            predicted_positive = np.sum(y_pred == label)
            precision = true_positive / predicted_positive if predicted_positive > 0 else 0
            precisions.append(precision)

        if average == 'macro':
            return np.mean(precisions)
        elif average == 'micro':
            true_positive_total = np.sum(y_true == y_pred)
            predicted_positive_total = len(y_pred)
            return true_positive_total / predicted_positive_total if predicted_positive_total > 0 else 0

    @staticmethod
    def recall(y_true, y_pred, average='macro'):
        unique_labels = np.unique(y_true)
        recalls = []

        for label in unique_labels:
            true_positive = np.sum((y_true == label) & (y_pred == label))
            actual_positive = np.sum(y_true == label)
            recall = true_positive / actual_positive if actual_positive > 0 else 0
            recalls.append(recall)

        if average == 'macro':
            return np.mean(recalls)
        elif average == 'micro':
            true_positive_total = np.sum(y_true == y_pred)
            actual_positive_total = len(y_true)
            return true_positive_total / actual_positive_total if actual_positive_total > 0 else 0

    @staticmethod
    def f1_score(y_true, y_pred, average='macro'):
        precision = Metrics.precision(y_true, y_pred, average=average)
        recall = Metrics.recall(y_true, y_pred, average=average)
        
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)