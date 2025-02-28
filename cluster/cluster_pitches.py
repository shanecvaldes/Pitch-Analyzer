from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
import skfuzzy as fuzz
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

def pca_data(X, n_components=2):
    # print(X.shape)
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)

def standardize(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def fuzzy_k_means(data, pitch_type, data_name):
    n_clusters = 3
    os.makedirs('./graphics/fuzzy_k_means/', exist_ok=True)
    os.makedirs(f'./graphics/fuzzy_k_means/{pitch_type}/', exist_ok=True)

    # Transpose the data for cmeans()
    cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
        data.T, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None
    )

    hard_clusters = np.argmax(u, axis=0)
    scatter_helper_plotly(data, hard_clusters, f'./graphics/fuzzy_k_means/{pitch_type}/{data_name}', cntr)

    '''
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(n_clusters):
        ax.scatter(data[:, 0], data[:, 1], c=u[i], alpha=0.6, label=f'Cluster {i+1}')

    # Plot cluster centers correctly
    ax.scatter(cntr[:, 0], cntr[:, 1], c='red', marker='X', s=200, label='Cluster Centers')

    ax.set_title('Fuzzy C-Means Clustering')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()
    plt.show()
    '''

def k_means(data, pitch_type, data_name, num_iterations=100):
    os.makedirs('./graphics/k_means/', exist_ok=True)
    os.makedirs(f'./graphics/k_means/{pitch_type}', exist_ok=True)
    tensor_data = torch.from_numpy(data).float()
    k = 4
    
    centroids = tensor_data[torch.randperm(tensor_data.size(0))[:k]]

    for _ in range(num_iterations):
        distances = torch.cdist(tensor_data, centroids)
        
        _, labels = torch.min(distances, dim=1)

        for i in range(k):
            if  torch.sum(labels==i) > 0:
                centroids[i] = torch.mean(tensor_data[labels==i], dim=0)

    scatter_helper_plotly(data, labels=labels.numpy(), path=f'./graphics/k_means/{pitch_type}/{data_name}_{num_iterations}', means=centroids.numpy())

    # old matplotlib code
    '''plt.scatter(data[:, 0], data[:, 1], c=labels.numpy(), cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, color='red')
    plt.show()'''

def gaussian_mix(data, pitch_type, data_name):
    n_components = 3
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(data)
    labels = gmm.predict(data)

    os.makedirs('./graphics/gaussian/', exist_ok=True)
    os.makedirs(f'./graphics/gaussian/{pitch_type}', exist_ok=True)

    means = gmm.means_
    scatter_helper_plotly(data=data, labels=labels, path=f'./graphics/gaussian/{pitch_type}/{data_name}', means = means)
    
    # old matplotlib code
    '''plt.figure(figsize=(8, 6))
    colors = plt.colormaps.get_cmap("tab10")  # Dynamic colormap
    for i in range(n_components):
        cluster_points = data[labels==i]
        plt.scatter(cluster_points[:,0], cluster_points[:,1],
                    label=f'Cluster{i}', color = colors(i), alpha=0.6, edgecolors='k')
        # Plot GMM cluster centers
        plt.scatter(means[i, 0], means[i, 1], 
                    marker='X', s=200, color=colors(i), edgecolors='black', label=f'Center {i}')
    plt.title(f'Gaussian Mixture Model Clustering (n={n_components})')
    plt.legend()
    plt.show()'''

def dbscan(data, pitch_type, data_name):
    os.makedirs('./graphics/db_scan/', exist_ok=True)
    os.makedirs(f'./graphics/db_scan/{pitch_type}', exist_ok=True)
    db = DBSCAN(eps=0.5, min_samples=5).fit(data)
    # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    # core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    scatter_helper_plotly(data, labels, path=f'./graphics/db_scan/{pitch_type}/{data_name}')
    
    # old matplotlib code
    '''
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    unique_labels = set(labels)
    colors = plt.colormaps.get_cmap("tab10")  # Dynamic colormap
    for k in unique_labels:
        col = colors(k)
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)
        
        plt.scatter(data[class_member_mask, 0], 
                    data[class_member_mask, 1], 
                    c=[col], label=f'Cluster {k}' if k != -1 else 'Noise',
                    edgecolors='k', s=50)
    plt.title('number of clusters: %d' % n_clusters_)
    plt.show()'''

def hierarchical(data, pitch_type, data_name):
    methods = ['complete', 'ward', 'centroid', 'median', 'average', 'weighted']
    metrics = ['euclidean']
    hyperparam = [(method, metric) for method in methods for metric in metrics]

    os.makedirs('./graphics/hierarchical/', exist_ok=True)
    os.makedirs(f'./graphics/hierarchical/{pitch_type}', exist_ok=True)

    for method, metric in hyperparam:
        try:
            Z = linkage(data, method, metric)
            dendogram_helper_plotly(Z, f'./graphics/hierarchical/{pitch_type}/{data_name}_{metric}_{method}.html')
        except RecursionError:
            continue

def spectral(data, pitch_type, data_name):
    os.makedirs('./graphics/spectral/', exist_ok=True)
    os.makedirs(f'./graphics/spectral/{pitch_type}', exist_ok=True)
    spec_model = SpectralClustering(n_clusters=4, affinity='rbf', degree=4)
    labels_poly = spec_model.fit_predict(data)
    scatter_helper_plotly(data, labels_poly, f'./graphics/spectral/{pitch_type}/{data_name}')

    # old matplotlib code
    '''
    fig = px.scatter(temp_df, x='x', y='y', color='color')
    fig.show()
    plt.scatter(data[:,0], data[:,1], c=labels_poly, cmap='tab10')
    plt.title('Spectral clustering with Polynomial affinity')
    plt.show()
    '''
    


# region plot helper

def scatter_helper_plotly(data, labels, path, means=np.array([])):
    # print(set([i for i in labels]))
    opacity = 0.3 if means.size > 0 else 1  # Lower opacity if means exist

    if data.shape[1] == 3:
        df = pd.DataFrame({'x':data[:,0], 'y':data[:,1], 'z':data[:,2], 'color':labels})
        fig = px.scatter_3d(df, x='x', y='y', z='z', color='color', opacity=opacity)

        # if the means exist, scatter them too
        if means.size > 0:
            fig.add_trace(go.Scatter3d(
                x=means[:, 0], y=means[:, 1], z=means[:, 2],
                mode='markers',
                marker=dict(size=8, color='black', symbol='x'),
                name='Means'
            ))
            
        fig.write_html(f'{path}_3d_rendering.html')

    elif data.shape[1] == 2:
        df = pd.DataFrame({'x':data[:,0], 'y':data[:,1], 'color':labels})
        fig = px.scatter(df, x='x', y='y', color='color', opacity=opacity)

        # if the means exist, scatter them too
        if means.size > 0:
            fig.add_trace(go.Scatter(
                x=means[:, 0], y=means[:, 1],
                mode='markers',
                marker=dict(size=8, color='black', symbol='x'),
                name='Means'
            ))

        fig.write_html(f'{path}_2d_rendering.html')
    # fig.show()
    

def dendogram_helper_plotly(data, path):
    dend = dendrogram(data, p=8, truncate_mode='level', no_plot=True)
    trunc = np.column_stack([dend['icoord'], dend['dcoord']])
    fig = ff.create_dendrogram(trunc)
    fig.update_layout(width=1000, height=500)
    fig.write_html(f'{path}.html')

# endregion



def cluster_data(file_name):
    df = pd.read_csv(f'../stats/pitches_stuff_summarized/{file_name}', encoding='latin1')
    # columns_stuff = ['release_speed', 'pfx_x', 'pfx_z', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'release_spin_rate', 'effective_speed']
    columns_stuff = ['pfx_x', 'pfx_z', 'ax', 'ay', 'az', 'release_spin_rate', 'effective_speed']

    # original data
    original_df = df[columns_stuff].to_numpy()
    grouped_df = df.groupby(['pitcher']).mean().reset_index()[columns_stuff].to_numpy()

    if len(grouped_df) <= 3:
        print('Not enough pitchers for a dataset')
        return 

    # normalized data
    std_original_df = standardize(original_df)
    std_grouped_df = standardize(grouped_df)

    # pca reduced 2D data
    reduced_grouped_df = pca_data(grouped_df)
    reduced_original_df = pca_data(original_df)

    # pca reduced standard 2D data
    std_reduced_grouped_df = pca_data(std_grouped_df)
    std_reduced_original_df = pca_data(std_original_df)

    # pca reduced 3D data
    reduced_3d_grouped_df = pca_data(grouped_df, 3)
    reduced_3d_original_df = pca_data(original_df, 3)

    # pca reduced standard 3D data
    std_reduced_3d_grouped_df = pca_data(std_grouped_df, 3)
    std_reduced_3d_original_df = pca_data(std_original_df, 3)

    os.makedirs('./graphics', exist_ok=True)
    pitch_type = file_name.removesuffix('.csv')

    # hierachical data
    name = ['std_original_df', 'std_grouped_df', 
            'reduced_grouped_df', 'reduced_original_df', 
            'std_reduced_grouped_df', 'std_reduced_original_df',
            'reduced_3d_grouped_df', 'reduced_3d_original_df',
            'std_reduced_3d_grouped_df', 'std_reduced_3d_original_df']
    all_data = [std_original_df, std_grouped_df, 
                reduced_grouped_df, reduced_original_df, 
                std_reduced_grouped_df, std_reduced_original_df,
                reduced_3d_grouped_df, reduced_3d_original_df,
                std_reduced_3d_grouped_df, std_reduced_3d_original_df]

    for data_name, data in zip(name, all_data):
        hierarchical(data, pitch_type, data_name)


    name = ['std_reduced_grouped_df', 'std_reduced_original_df',
            'std_reduced_3d_grouped_df', 'std_reduced_3d_original_df']
    all_data = [std_reduced_grouped_df, std_reduced_original_df,
                std_reduced_3d_grouped_df, std_reduced_3d_original_df]
    # only use 2D and 3D data for the others to allow for graphing
    for data_name, data in zip(name, all_data):
        gaussian_mix(data, pitch_type, data_name)
        spectral(data, pitch_type, data_name)
        dbscan(data, pitch_type, data_name)
        k_means(data, pitch_type, data_name)
        fuzzy_k_means(data, pitch_type, data_name)

    


def main():
    os.makedirs('/graphics', exist_ok=True)
    # print(temp)
    # cluster_data(f'stuff_summarized_FF.csv')
    for pitch_file in os.listdir(f'../stats/pitches_stuff_summarized'):
        print(pitch_file)
        cluster_data(pitch_file)

main()