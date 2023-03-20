import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
data = pd.read_csv('customer_data.csv')

# Chuẩn bị dữ liệu cho K-means
X = data.iloc[:, [2, 3]].values

# Tìm số lượng cluster tối ưu bằng phương pháp elbow
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Phương pháp Elbow')
plt.xlabel('Số lượng cluster')
plt.ylabel('WCSS')
plt.show()

# Áp dụng K-means với số lượng cluster được chọn
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# Hiển thị kết quả phân loại
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Phân loại khách hàng')
plt.xlabel('Số lượng sản phẩm mua')
plt.ylabel('Giá trị đơn hàng')
plt.legend()
plt.show()