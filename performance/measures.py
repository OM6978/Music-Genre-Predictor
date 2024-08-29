import numpy as np

class measures:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def precision(self, average='macro'):
        tp = np.sum(self.y_true * self.y_pred, axis=0)
        fp = np.sum(self.y_pred * (1 - self.y_true), axis=0)
        precision = tp / (tp + fp + 1e-10)

        if average == 'micro':
            return np.mean(precision)
        else:
            return np.mean(precision[precision != np.inf])

    def recall(self, average='macro'):
        tp = np.sum(self.y_true * self.y_pred, axis=0)
        fn = np.sum((1 - self.y_pred) * self.y_true, axis=0)
        recall = tp / (tp + fn + 1e-10)

        if average == 'micro':
            return np.mean(recall)
        else:
            return np.mean(recall[recall != np.inf])

    def f1_score(self, average='macro'):
        precision = self.precision(average)
        recall = self.recall(average)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        return f1

    def accuracy(self):
        correct_predictions = 0
        for i in range(len(self.y_pred)):
            correct_predictions = correct_predictions + int(self.y_pred[i] == self.y_true[i])
        
        total_samples = len(self.y_true)
        accuracy = correct_predictions / total_samples
        
        return accuracy