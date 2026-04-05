import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)

# 加载红酒质量数据集
wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(wine_url, sep=';')

# 数据预处理
X = df.drop('quality', axis=1).values
y_reg = df['quality'].values
y_cls = (df['quality'] > 6).astype(int).values


# 特征标准化
def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std, mean, std


X_std, X_mean, X_std_dev = standardize(X)
X_b = np.c_[np.ones((X_std.shape[0], 1)), X_std]


# 拆分训练集测试集
def train_test_split(X, y, test_ratio=0.2):
    m = X.shape[0]
    shuffled_indices = np.random.permutation(m)
    test_size = int(m * test_ratio)
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_b, y_reg)


# 分层拆分（逻辑回归专用）
def stratified_train_test_split(X, y, test_ratio=0.2):
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]

    test_pos_size = int(len(pos_indices) * test_ratio)
    test_neg_size = int(len(neg_indices) * test_ratio)

    np.random.shuffle(pos_indices)
    np.random.shuffle(neg_indices)

    test_indices = np.concatenate([pos_indices[:test_pos_size], neg_indices[:test_neg_size]])
    train_indices = np.concatenate([pos_indices[test_pos_size:], neg_indices[test_neg_size:]])

    np.random.shuffle(test_indices)
    np.random.shuffle(train_indices)
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


X_train_cls, X_test_cls, y_train_cls, y_test_cls = stratified_train_test_split(X_b, y_cls)


# 线性回归实现
class LinearRegression:
    def __init__(self):
        self.theta = None

    def fit(self, X_train, y_train):
        XTX = X_train.T @ X_train
        XTX_inv = np.linalg.inv(XTX + 1e-6 * np.eye(XTX.shape[0]))
        self.theta = XTX_inv @ X_train.T @ y_train

    def predict(self, X):
        return X @ self.theta


# 线性回归训练预测
lr_model = LinearRegression()
lr_model.fit(X_train_reg, y_train_reg)
y_pred_reg = lr_model.predict(X_test_reg)


# 逻辑回归实现
class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=10000):
        self.lr = lr
        self.n_iters = n_iters
        self.theta = None
        self.loss_history = []

    # Sigmoid函数（数值稳定）
    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    # 二元交叉熵损失
    def _binary_cross_entropy(self, y_true, y_hat):
        m = len(y_true)
        epsilon = 1e-10
        loss = -1 / m * np.sum(y_true * np.log(y_hat + epsilon) + (1 - y_true) * np.log(1 - y_hat + epsilon))
        return loss

    def fit(self, X_train, y_train):
        m, n = X_train.shape
        self.theta = np.zeros(n)

        for i in range(self.n_iters):
            z = X_train @ self.theta
            y_hat = self._sigmoid(z)
            grad = (1 / m) * X_train.T @ (y_hat - y_train)
            self.theta -= self.lr * grad

            loss = self._binary_cross_entropy(y_train, y_hat)
            self.loss_history.append(loss)

    def predict_proba(self, X):
        z = X @ self.theta
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)


# 逻辑回归训练预测
lr_cls_model = LogisticRegression(lr=0.01, n_iters=10000)
lr_cls_model.fit(X_train_cls, y_train_cls)
y_pred_cls = lr_cls_model.predict(X_test_cls)
y_pred_proba_cls = lr_cls_model.predict_proba(X_test_cls)

# 损失曲线可视化
plt.figure(figsize=(8, 4))
plt.plot(lr_cls_model.loss_history)
plt.xlabel("迭代次数")
plt.ylabel("二元交叉熵损失")
plt.title("逻辑回归训练损失下降曲线")
plt.show()


# 线性回归评估指标
def regression_metrics(y_true, y_pred):
    m = len(y_true)
    mse = np.sum((y_true - y_pred) ** 2) / m
    mae = np.sum(np.abs(y_true - y_pred)) / m
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return {"MSE": mse, "MAE": mae, "R²": r2}


reg_metrics = regression_metrics(y_test_reg, y_pred_reg)
print("=" * 30)
print("线性回归测试集评估结果:")
for k, v in reg_metrics.items():
    print(f"{k}: {v:.4f}")
print("=" * 30)


# 逻辑回归评估指标
def classification_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    return {
        "混淆矩阵": [[tn, fp], [fn, tp]],
        "准确率Accuracy": accuracy,
        "精确率Precision": precision,
        "召回率Recall": recall,
        "F1值": f1
    }


cls_metrics = classification_metrics(y_test_cls, y_pred_cls)
print("逻辑回归测试集评估结果:")
for k, v in cls_metrics.items():
    if k == "混淆矩阵":
        print(f"{k}:\n{v[0]}\n{v[1]}")
    else:
        print(f"{k}: {v:.4f}")
print("=" * 30)

# 加载Iris数据集
iris_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
iris_df = pd.read_csv(iris_url, header=None,
                      names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label'])

X_iris = iris_df.drop('label', axis=1).values
y_iris = iris_df['label'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}).values
X_iris_std, _, _ = standardize(X_iris)


# K-Means实现
class KMeans:
    def __init__(self, n_clusters=3, max_iters=1000, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels = None
        self.sse = None

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def fit(self, X):
        m, n = X.shape
        random_indices = np.random.choice(m, self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iters):
            self.labels = np.array([np.argmin([self._euclidean_distance(x, c) for c in self.centroids]) for x in X])

            new_centroids = np.zeros((self.n_clusters, n))
            for k in range(self.n_clusters):
                cluster_samples = X[self.labels == k]
                if len(cluster_samples) > 0:
                    new_centroids[k] = np.mean(cluster_samples, axis=0)
                else:
                    new_centroids[k] = self.centroids[k]

            centroid_change = np.sum(np.abs(new_centroids - self.centroids))
            if centroid_change < self.tol:
                print(f"K-Means在第{i + 1}次迭代收敛")
                break

            self.centroids = new_centroids

        self.sse = np.sum([self._euclidean_distance(x, self.centroids[label]) ** 2 for x, label in zip(X, self.labels)])

    def predict(self, X):
        return np.array([np.argmin([self._euclidean_distance(x, c) for c in self.centroids]) for x in X])


# K-Means训练
np.random.seed(42)
kmeans_model = KMeans(n_clusters=3)
kmeans_model.fit(X_iris_std)
cluster_labels = kmeans_model.labels


# 聚类评估（兰德指数）
def rand_index(y_true, y_pred):
    m = len(y_true)
    tp_tn = 0
    total_pairs = m * (m - 1) / 2
    for i in range(m):
        for j in range(i + 1, m):
            same_true = (y_true[i] == y_true[j])
            same_pred = (y_pred[i] == y_pred[j])
            if same_true == same_pred:
                tp_tn += 1
    return tp_tn / total_pairs


ri = rand_index(y_iris, cluster_labels)
print("=" * 30)
print("K-Means聚类评估结果:")
print(f"簇内平方和SSE: {kmeans_model.sse:.4f}")
print(f"兰德指数RI: {ri:.4f}")
print("=" * 30)


# PCA降维（2维可视化）
def pca(X, n_components=2):
    cov_matrix = np.cov(X.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_eigenvectors = eigenvectors[:, sorted_indices[:n_components]]
    X_pca = X @ top_eigenvectors
    return X_pca


X_iris_pca = pca(X_iris_std)

# 聚类结果可视化
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
scatter1 = plt.scatter(X_iris_pca[:, 0], X_iris_pca[:, 1], c=cluster_labels, cmap='viridis', s=50)
pca_eigenvectors = np.linalg.eig(np.cov(X_iris_std.T))[1][:, :2]
centroids_pca = kmeans_model.centroids @ pca_eigenvectors
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='X', s=200, label='聚类中心')
plt.title("K-Means聚类结果")
plt.legend(handles=scatter1.legend_elements()[0] + [
    plt.Line2D([], [], marker='X', color='red', linestyle='None', markersize=10)],
           labels=['簇0', '簇1', '簇2', '聚类中心'])

plt.subplot(1, 2, 2)
scatter2 = plt.scatter(X_iris_pca[:, 0], X_iris_pca[:, 1], c=y_iris, cmap='viridis', s=50)
plt.title("Iris真实标签")
plt.legend(handles=scatter2.legend_elements()[0], labels=['Setosa', 'Versicolor', 'Virginica'])

plt.tight_layout()
plt.show()