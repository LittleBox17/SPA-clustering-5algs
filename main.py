# author: hsl
# 2023/5/9 14:43
import matplotlib.pyplot as plt
from clusterAlgs5 import Clustering

# 数据
dir = "./traceSet/"
dataname = "sm2"

# 5种基本聚类
cluster5 = Clustering()
cluster5.set_directory(dir)
cluster5.set_dataset(dataname)
res = cluster5.run_cluster5(
    n_iterations=5,
    pca_dim=2,
    visualization=True,
    vistype='cerrs' # 'cres'：聚类结果； 'cerrs':标记错误
)
plt.show()
