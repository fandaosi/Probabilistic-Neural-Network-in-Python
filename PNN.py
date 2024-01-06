import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import KFold

class PNN:
    def __init__(self, sigma):
        self.sigma = sigma
        self.class_index = None
        self.X = None

    def pattern_layer(self, X):
        # return np.exp(- cdist(X, self.X) ** 2 / (2 * (self.sigma ** 2)))
        return rbf_kernel(X, self.X, gamma=1 / (2 * (self.sigma ** 2)))

    def summation_layer(self, pattern):
        summation = np.zeros((pattern.shape[0], self.class_index.shape[0]))
        for i, index in enumerate(self.class_index):
            summation[:, i] = pattern[:, index].sum(axis=1)
        return summation / summation.sum(axis=1, keepdims=True)

    def output_layer(self, summation):
        return np.argmax(summation, axis=1)

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        self.class_index = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray().astype(np.bool_).T
        self.X = X

    def predict(self, X):
        pattern = self.pattern_layer(X)
        summation = self.summation_layer(pattern)
        output = self.output_layer(summation)
        return output


if __name__ == '__main__':
    iris = load_iris()
    X, y = MinMaxScaler().fit_transform(iris.data), iris.target
    acc_list = list()

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        pnn = PNN(0.1)
        pnn.fit(X_train, y_train)
        acc = accuracy_score(y_test, pnn.predict(X_test))
        acc_list.append(acc)
        print('%d accuracy: %.2f' % (i, acc))
    print('mean accuracy: %.2f' % np.mean(acc_list))