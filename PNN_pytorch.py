import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.nn import functional as F


class PatternLayer(nn.Module):
    def __init__(self, X, sigma):
        super().__init__()
        self.X = X
        self.sigma = sigma

    def forward(self, X):
        return torch.exp(-torch.cdist(X, self.X) ** 2 / (2 * (self.sigma ** 2)))


class SummationLayer(nn.Module):
    def __init__(self, class_index):
        super().__init__()
        self.class_index = class_index

    def forward(self, pattern):
        summation = torch.zeros((pattern.shape[0], self.class_index.shape[0]))
        for i, index in enumerate(self.class_index):
            summation[:, i] = pattern[:, index].sum(dim=1)
        return summation / summation.sum(dim=1, keepdim=True)


class OutputLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, summation):
        return torch.argmax(summation, dim=1)


class PNN(nn.Module):
    def __init__(self, X, y, sigma):
        super().__init__()
        class_index = F.one_hot(y.to(torch.int64)).to(torch.bool).T
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        X = X.to(self.device)
        self.pattern_layer = PatternLayer(X, sigma)
        self.summation_layer = SummationLayer(class_index)
        self.output_layer = OutputLayer()

    def forward(self, X):
        X = X.to(self.device)
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
        X_train, y_train = torch.as_tensor(X[train_index]), torch.as_tensor(y[train_index])
        X_test, y_test = torch.as_tensor(X[test_index]), torch.as_tensor(y[test_index])
        pnn = PNN(X_train, y_train, 0.1)
        acc = accuracy_score(y_test, pnn.forward(X_test))
        acc_list.append(acc)
        print('%d accuracy: %.2f' % (i, acc))
    print('mean accuracy: %.2f' % np.mean(acc_list))