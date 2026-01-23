import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("clusters.csv")

points = data[data["type"] == 0]
centroids = data[data["type"] == 1]


plt.scatter(
    points["x"], points["y"], 
    c=points["cluster"], 
    s=8, 
    cmap="viridis"
)
plt.scatter(
    centroids["x"], centroids["y"],
    s=10,
    c='black',
    linewidths=2,
    label="Centroids"
)


plt.title("K-Means Clustering Visualization")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()