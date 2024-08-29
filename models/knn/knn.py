import numpy as np

class knn:
    def euclidean_dist(self,point):
        return np.sum((point[self.mask] - self.data[:,self.mask])**2, axis=1)**0.5
    
    def manhattan_dist(self,point):
        return np.sum(abs(point - self.data), axis=1)
    
    def test_data(self,X,K,distance):
        y_pred = np.zeros(X.shape[0], dtype=self.data[:,-1].dtype)
        # count = 0

        X = X[:,self.mask]
        for i, x in enumerate(X):
            # count += 1
            # print("Prediction number ", count)

            if distance == 'manhattan':
                distances = self.manhattan_dist(x)
            elif distance == 'euclidean':
                distances = self.euclidean_dist(x)
            
            k_indices = np.argsort(distances)[:K]

            k_nearest_labels = self.y[k_indices]
            labels, counts = np.unique(k_nearest_labels, return_counts=True)

            y_pred[i] = labels[np.argmax(counts)]
        
        return y_pred

    def __init__(self,data):
        self.mask = np.ones(data.shape[1], dtype=bool)
        self.mask[-1] = False

        self.data = data[:,self.mask]
        self.y = data[:,-1]