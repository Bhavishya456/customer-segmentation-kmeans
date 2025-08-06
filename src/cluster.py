import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os

def load_data(p):
    return pd.read_csv(p)

def preprocess(d, f):
    X = d[f].copy()
    s = StandardScaler()
    return s.fit_transform(X), s

def get_scores(X, m=10):
    w, s = [], []
    for k in range(2, m+1):
        km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        lbl = km.fit_predict(X)
        w.append(km.inertia_)
        s.append(silhouette_score(X, lbl))
    return w, s

def plot_scores(w, s):
    fig, a1 = plt.subplots()
    a2 = a1.twinx()
    a1.plot(range(2, len(w)+2), w, 'bx-')
    a2.plot(range(2, len(s)+2), s, 'ro-')
    a1.set_xlabel('k')
    a1.set_ylabel('WCSS')
    a2.set_ylabel('Silhouette')
    plt.title('Cluster Analysis')
    plt.show()

def train(X, k):
    m = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    l = m.fit_predict(X)
    return m, l

def plot_clusters(d, f, l, c):
    d['Cluster'] = l
    sns.scatterplot(x=f[0], y=f[1], hue='Cluster', palette='tab10', data=d, s=60)
    plt.scatter(c[:, 0], c[:, 1], s=300, c='red', marker='X')
    plt.xlabel(f[0])
    plt.ylabel(f[1])
    plt.title('Customer Segmentation')
    plt.legend()
    plt.show()

def summary(d, f):
    return pd.concat([d['Cluster'].value_counts().sort_index().rename('Count'),
                      d.groupby('Cluster')[f].mean().round(2)], axis=1)

def main():
    data_path = os.path.join('data', 'Mall_Customers.csv')
    df = load_data(data_path)
    f = ['Annual Income (k$)', 'Spending Score (1-100)']
    X, _ = preprocess(df, f)
    w, s = get_scores(X, m=10)
    plot_scores(w, s)
    k = 5
    m, l = train(X, k)
    plot_clusters(df, f, l, m.cluster_centers_)
    print(summary(df, f))
    os.makedirs('outputs', exist_ok=True)
    df.to_csv(os.path.join('outputs', 'Mall_Customers_segmented.csv'), index=False)

if __name__ == '__main__':
    main()
