import numpy as np
# import scipt as sp
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt


def k_init(X, k):
    n, d = len(X), len(X[0])
    C = np.zeros([k, d])
    C[0] = X[np.random.choice(n)]

    for i in range(1, k):
        dists = [np.min([np.dot(x - C[j], x - C[j]) for j in range(i)]) for x in X]
        total_dists = sum(dists)
        probs = [d / total_dists for d in dists]
        idx = np.random.choice(n, p=probs)
        C[i] = X[idx]

    return C


def k_means_pp(X, k, max_iter):
    c0 = k_init(X, k)
    c = [c0]
    a = []
    objs = [compute_objective(X, c0)]

    for t in range(max_iter):
        dmap = assign_data2clusters(X, c[t])
        a.append(dmap)
        new_c = get_new_clusters(X, a[t])
        c.append(new_c)
        obj = compute_objective(X, c[t+1])
        objs.append(obj)

    return c0, c[max_iter], objs


def get_new_clusters(X, data_map):
    n, k, d = len(X), len(data_map[0]), len(X[0])
    clusters = {i: [] for i in range(k)}
    for i, a in enumerate(data_map):
        idx = list(a).index(1)
        clusters[idx].append(X[i])
    cnew = [np.mean(clusters[i], axis=0) for i in clusters]
    return np.array(cnew)


def assign_data2clusters(X, C):
    data_map = []
    for x in X:
        dists = get_dist_to_centers(x, C)
        k_assign = np.argmin(dists)
        assign = [1 if i == k_assign else 0 for i in range(len(C))]
        data_map.append(assign)
    return np.array(data_map)


def compute_objective(X, C):
    obj = 0
    for x in X:
        dists = get_dist_to_centers(x, C)
        cost = np.min(dists)
        obj += cost

    return obj


def get_dist_to_centers(x, C):
    dists = []
    for c in C:
        dist = np.dot(x - c, x - c)
        dists.append(dist)
    return np.array(dists)


def plot_clusters(X, y, label_names, C, C0):
    x1, x2 = X.T[0], X.T[1]
    c1, c2 = C.T[0], C.T[1]
    c10, c20 = C0.T[0], C0.T[1]
    df = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'labels': y
    })

    for i in range(len(label_names)):
        dfl = df[df['labels'] == i]
        plt.scatter(dfl['x1'], dfl['x2'])

    plt.scatter(c1, c2, marker='x')
    plt.show()


def plot_obj(obj, k):
    plt.title(f'Objective vs. Iterations for {k} Clusters')
    plt.ylabel('f')
    plt.xlabel('iter')
    plt.plot(list(range(len(obj))), obj)
    plt.show()


def plot_acc(X, labels, krange):
    objs = []
    for k in krange:
        corig, c, objs2 = k_means_pp(X, k, 50)
        obj = compute_objective(X, c)
        objs.append(obj)

    plt.title('Objective Loss vs. K Clusters')
    plt.ylabel('f')
    plt.xlabel('K')
    plt.plot(krange, objs)
    plt.show()


if __name__ == "__main__":
    iris_data = datasets.load_iris()
    f1, y, keys1, labels = iris_data.data.T, iris_data.target, iris_data.feature_names, iris_data.target_names

    keys2 = ['sepal l/w', 'petal l/w']
    data = np.array([
        np.divide(f1[0], f1[1]),
        np.divide(f1[2], f1[3])
    ]).T

    plot_acc(data, y, range(1, 6))

    K = 3
    ci, cf, objs1 = k_means_pp(data, K, 50)
    plot_clusters(data, y, labels, cf, ci)
    plot_obj(objs1, K)


