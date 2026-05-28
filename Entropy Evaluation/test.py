import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

# 加载数据
x = np.load('centroids_left.npz', allow_pickle=True)
centroids = x['x']
centroid_labels = x['y']

# 将标签转换为整数编码
label_encoder = LabelEncoder()
centroid_labels_encoded = label_encoder.fit_transform(centroid_labels)

# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2, random_state=42, init='pca')
centroids_2d = tsne.fit_transform(centroids)

# 绘制散点图
plt.figure(figsize=(10, 8))
scatter = plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c=centroid_labels_encoded, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='User Label')
plt.title('t-SNE Visualization of User Centroids')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()