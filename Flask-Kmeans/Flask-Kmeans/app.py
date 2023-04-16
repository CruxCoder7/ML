import numpy as np
import io
from flask import Flask, request, Response, render_template
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd


app = Flask(__name__)

df = pd.read_csv(r"C:\Users\91755\Downloads\data_new.csv")
data = df.iloc[:, 2:].values

# sample dataset
X = np.array([[1, 1], [1, 2], [2, 2], [4, 5], [5, 5],
             [5, 6], [6, 6], [8, 7], [7, 8], [9, 9]])


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route('/cluster', methods=['POST'])
def cluster():

    k = int(request.form['k'])
    km = KMeans(n_clusters=k, random_state=0)
    km.fit(data)
    labels = km.labels_
    centroids = km.cluster_centers_
    y = km.predict(data)
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[y == 0, 0], data[y == 0, 1], data[y == 0, 2],
               s=40, color='blue', rlabel="cluster 0")
    ax.scatter(data[y == 1, 0], data[y == 1, 1], data[y == 1, 2],
               s=40, color='orange', label="cluster 1")
    ax.scatter(data[y == 2, 0], data[y == 2, 1], data[y == 2, 2],
               s=40, color='green', label="cluster 2")
    ax.scatter(data[y == 3, 0], data[y == 3, 1],
               data[y == 3, 2], s=40, color='red', label="cluster 3")
    ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[
               :, 1], km.cluster_centers_[:, 2], color='black', s=150, label='centroids')
    ax.set_xlabel('Age ')
    ax.set_ylabel('Anual Income')
    ax.set_zlabel('Spending Score')
    ax.legend()
    output = io.BytesIO()
    fig.savefig(output)
    # fig, ax = plt.subplots()

    # ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    # ax.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=300, c='r')
    # ax.set_title('K-means clustering')
    # ax.set_xlabel('Feature 1')
    # ax.set_ylabel('Feature 2')

    # output = io.BytesIO()
    # fig.savefig(output)

    return Response(output.getvalue(), mimetype='image/png')


if __name__ == "__main__":
    app.run(debug=True)
