from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
import skfuzzy as fuzz
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
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

# use global arrays to store the names of the pitchers
original_df_names = []
grouped_df_names = []
original_years = []

def cluster_dec(func):
    def cluster_wrapper(data_combined, pitch_type, src_folder):
        for data_name, data in data_combined.items():
            func(data, pitch_type, data_name, src_folder)
    return cluster_wrapper


# region pre data processors

def pca_data(X, n_components=2):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)

def standardize(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

# endregion

# region algorithms

@cluster_dec
def fuzzy_k_means(data, pitch_type, data_name, src_folder):
    n_clusters = 3

    # Transpose the data for cmeans()
    cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
        data.T, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None
    )

    hard_clusters = np.argmax(u, axis=0)
    graphic_path, label_path = generate_path('fuzzy_k_means', data, data_name, pitch_type, src_folder)

    graphic_path += f'/{n_clusters}c.html'
    label_path += f'/{n_clusters}c.csv'

    scatter_helper_plotly(data, hard_clusters, graphic_path, label_path, cntr)

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

@cluster_dec
def k_means(data, pitch_type, data_name, src_folder, num_iterations=100):
    tensor_data = torch.from_numpy(data).float()
    k = 4
    
    centroids = tensor_data[torch.randperm(tensor_data.size(0))[:k]]

    for _ in range(num_iterations):
        distances = torch.cdist(tensor_data, centroids)
        
        _, labels = torch.min(distances, dim=1)

        for i in range(k):
            if  torch.sum(labels==i) > 0:
                centroids[i] = torch.mean(tensor_data[labels==i], dim=0)


    graphic_path, label_path = generate_path('k_means', data, data_name, pitch_type, src_folder)

    graphic_path += f'/{k}c.html'
    label_path += f'/{k}c.csv'

    scatter_helper_plotly(data, labels=labels.numpy(), graphic_path=graphic_path, label_path=label_path, means=centroids.numpy())

    # old matplotlib code
    '''plt.scatter(data[:, 0], data[:, 1], c=labels.numpy(), cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, color='red')
    plt.show()'''

@cluster_dec
def gaussian_mix(data, pitch_type, data_name, src_folder):
    n_components = 3
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(data)
    labels = gmm.predict(data)

    graphic_path, label_path = generate_path('gaussian_mix', data, data_name, pitch_type, src_folder)

    graphic_path += f'/{n_components}c.html'
    label_path += f'/{n_components}c.csv'

    means = gmm.means_
    scatter_helper_plotly(data=data, labels=labels, graphic_path=graphic_path, label_path=label_path, means = means)
    
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

@cluster_dec
def dbscan(data, pitch_type, data_name, src_folder):
    eps = 0.5
    graphic_path, label_path = generate_path('dbscan', data, data_name, pitch_type, src_folder)

    graphic_path += f'/{eps}eps.html'
    label_path += f'/{eps}eps.csv'

    db = DBSCAN(eps=eps, min_samples=5).fit(data)
    # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    # core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    scatter_helper_plotly(data, labels, graphic_path, label_path)
    
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

@cluster_dec
def hierarchical(data, pitch_type, data_name, src_folder):
    methods = ['complete', 'ward', 'centroid', 'median', 'average', 'weighted']
    metrics = ['euclidean']
    hyperparam = [(method, metric) for method in methods for metric in metrics]
    graphic_path, label_path = generate_path('hierarchical', data, data_name, pitch_type, src_folder)

    global original_df_names
    global grouped_df_names
    if 'grouped' in graphic_path:
        names = grouped_df_names
    else:
        names = original_df_names

    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        print("Data contains NaN or INF values")

    data = np.asarray(data)

    for method, metric in hyperparam:

        try:
            Z = linkage(data, method, metric)
            # print(f'Shape of linkage: {Z.shape}')
            removed_label, removed_indexes = track_labels(data, Z, graphic_path)

            labels = fcluster(Z, 10, criterion='distance')

            scatter_helper_plotly(data, labels, graphic_path + f'/{metric}_{method}.html', label_path + f'/{metric}_{method}.csv')

        except RecursionError:
            continue

@cluster_dec
def spectral(data, pitch_type, data_name, src_folder):
    n_clusters = 4
    graphic_path, label_path = generate_path('spectral', data, data_name, pitch_type, src_folder)

    graphic_path += f'/{n_clusters}c.html'
    label_path += f'/{n_clusters}c.csv'

    spec_model = SpectralClustering(n_clusters=n_clusters, affinity='rbf', degree=4)
    labels_poly = spec_model.fit_predict(data)
    scatter_helper_plotly(data, labels_poly, graphic_path, label_path)

    # old matplotlib code
    '''
    fig = px.scatter(temp_df, x='x', y='y', color='color')
    fig.show()
    plt.scatter(data[:,0], data[:,1], c=labels_poly, cmap='tab10')
    plt.title('Spectral clustering with Polynomial affinity')
    plt.show()
    '''
    
# endregion

# region plot helper

def scatter_helper_plotly(data, labels, graphic_path, label_path, means=np.array([])):
    opacity = 0.3 if means.size > 0 else 1  # Lower opacity if means exist

    global original_df_names
    global grouped_df_names
    global original_years

    '''js_flag = 'cdn'
    if os.path.exists(f'{path}_3d_rendering.html'):
        js_flag = False'''
    year_flag = False
    if 'grouped' in graphic_path:
        names = grouped_df_names
    else:
        year_flag = True
        names = original_df_names
    label_df = pd.DataFrame({})
    if year_flag:
        label_df['game_year'] = original_years
    label_df['name'] = names
    label_df['label'] = labels
    
    label_df.to_csv(label_path, index=False)

    # print(f'Shape of the data: {data.shape}')
    # print(f'Shape of the labels: {labels.shape}')
    # print(f'Shape of the names: {names.shape}')


    if data.shape[1] == 3:
        df = pd.DataFrame({'x':data[:,0], 'y':data[:,1], 'z':data[:,2], 'color':labels, 'name':names})
        df['color'] = df['color'].astype(str)

        fig = px.scatter_3d(df, x='x', y='y', z='z', color='color', hover_name = 'name', opacity=opacity)

        # if the means exist, scatter them too
        if means.size > 0:
            fig.add_trace(go.Scatter3d(
                x=means[:, 0], y=means[:, 1], z=means[:, 2],
                mode='markers',
                marker=dict(size=8, color='black', symbol='x'),
                name='Means'
            ))
        
        '''with open(f'{graphic_path}', 'a') as f:
            f.write(fig.to_html(full_html=False, include_plotlyjs=js_flag))'''
        fig.write_html(f'{graphic_path}', full_html=True)

    elif data.shape[1] == 2:
        df = pd.DataFrame({'x':data[:,0], 'y':data[:,1], 'color':labels, 'name':names})
        df['color'] = df['color'].astype(str)

        fig = px.scatter(df, x='x', y='y', color='color', hover_name = 'name', opacity=opacity)
        # if the means exist, scatter them too
        if means.size > 0:
            fig.add_trace(go.Scatter(
                x=means[:, 0], y=means[:, 1],
                mode='markers',
                marker=dict(size=8, color='black', symbol='x'),
                name='Means'
            ))

        fig.write_html(f'{graphic_path}', full_html=True)
    # fig.show()

# DEAD FUNCTION
def heatmap_healper_plotly(data, Z, labels, graphic_path, label_path):
    dend = dendrogram(Z, no_plot=True)
    ordered_indices = dend['leaves']
    
    ordered_labels = [labels[i] for i in ordered_indices]
    new_data = pca_data(data.copy(), 2)

    # Reorder the data based on the dendrogram
    ordered_data = new_data[ordered_indices, :]
    fig = ff.create_annotated_heatmap(
        z=ordered_data,  # Data for the heatmap
        x=[f'Feature {i}' for i in range(1, ordered_data.shape[1] + 1)],  # Feature labels
        y=ordered_labels,  # Sample labels based on dendrogram ordering
        colorscale='Viridis',  # Color scale
        showscale=True  # Show color scale bar
    )
    fig.write_html(graphic_path)

# DEAD FUNCTION
def dendogram_helper_plotly(data, Z, labels, graphic_path, label_path):
    # broken with labels
    global original_df_names
    global grouped_df_names

    if 'grouped' in graphic_path:
        names = grouped_df_names
    else:
        names = original_df_names

    dend = dendrogram(data, p=6, truncate_mode='level', labels=names, no_plot=True)
    # trunc = np.column_stack([dend['icoord'], dend['dcoord']])
    # truncated_labels = dend['ivl']
    # truncated_labels = [names[i] for i in dend['leaves']]
    
    # fig = ff.create_dendrogram(trunc, labels=truncated_labels)
    fig = ff.create_dendrogram(Z)
    # fig.update_layout(width=1000, height=500)
    fig.write_html(graphic_path)

# endregion

def track_labels(data, Z, graphic_path):
    global original_df_names
    global grouped_df_names
    
    # Choose appropriate labels based on the graphic path
    if 'grouped' in graphic_path:
        names = grouped_df_names
    else:
        names = original_df_names

    # Track the original indices of the data samples
    num_samples = len(data)
    current_labels = names.copy()
    
    # Create a list to track which labels are associated with each cluster
    clusters = {i: [current_labels[i]] for i in range(num_samples)}  # Each sample starts as its own cluster
    
    # Track the removed labels
    removed_label = None

    # For each merge in the linkage matrix Z, merge two clusters
    for i, row in enumerate(Z):
        idx1, idx2 = int(row[0]), int(row[1])

        # Merge clusters
        merged_labels = clusters.pop(idx1, []) + clusters.pop(idx2, [])
        clusters[num_samples + i] = merged_labels  # Assign new cluster index

        # Track the first removed label (i.e., last one in the merged group)
        if removed_label is None:
            removed_label = merged_labels[-1]

    # print(f"Removed label: {removed_label}")
    # print(f"Shape of linkage matrix: {Z.shape}, Shape of data: {data.shape}")
    
    return removed_label, np.where(names == removed_label)[0]

def cluster_data(file_name, src_folder):
    global original_df_names
    global original_years
    global grouped_df_names
    print(f'Working on {src_folder} and {file_name}')
    df = pd.read_csv(f'../stats/{src_folder}/{file_name}', encoding='latin1')
    print(f'Total len of data: {len(df)}')
    # columns_stuff = ['release_speed', 'pfx_x', 'pfx_z', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'release_spin_rate', 'effective_speed']
    # columns_stuff = ['pfx_x', 'pfx_z', 'ax', 'ay', 'az', 'release_spin_rate', 'effective_speed']
    columns_stuff = df.columns.to_list()
    columns_stuff.remove('game_year')
    columns_stuff.remove('pitcher')

    # original data
    original_df_names, original_years = df['pitcher'].to_numpy(), df['game_year'].to_numpy()

    original_df = df[columns_stuff].to_numpy()
    grouped_df = df.groupby(['pitcher']).mean()#.reset_index()[columns_stuff].to_numpy()

    grouped_df_names = grouped_df.index.to_numpy()

    if len(grouped_df) <= 3:
        print('Not enough pitchers for a dataset')
        return 

    # normalized data
    std_original_df, std_grouped_df = standardize(original_df), standardize(grouped_df)

    # pca reduced 2D data
    reduced_grouped_df, reduced_original_df = pca_data(grouped_df), pca_data(original_df)

    # pca reduced standard 2D data
    std_reduced_grouped_df, std_reduced_original_df = pca_data(std_grouped_df), pca_data(std_original_df)

    # pca reduced 3D data
    reduced_3d_grouped_df, reduced_3d_original_df = pca_data(grouped_df, 3), pca_data(original_df, 3)

    # pca reduced standard 3D data
    std_reduced_3d_grouped_df, std_reduced_3d_original_df = pca_data(std_grouped_df, 3), pca_data(std_original_df, 3)

    os.makedirs('./graphics', exist_ok=True)
    os.makedirs('./labels', exist_ok=True)

    pitch_type = file_name.removesuffix('.csv')
    
    # hierachical data
    names = ['std_original_df', 'std_grouped_df', 
            'reduced_grouped_df', 'reduced_original_df', 
            'std_reduced_grouped_df', 'std_reduced_original_df',
            'reduced_3d_grouped_df', 'reduced_3d_original_df',
            'std_reduced_3d_grouped_df', 'std_reduced_3d_original_df',
            'original_df', 'grouped_df']
    
    data_zipped = {key:value for key, value in locals().items() if key in names}

    hierarchical(data_zipped, pitch_type, src_folder)

    names = ['std_reduced_grouped_df', 'std_reduced_original_df',
            'std_reduced_3d_grouped_df', 'std_reduced_3d_original_df']
    
    data_zipped = {key:value for key, value in locals().items() if key in names}

    spectral(data_zipped, pitch_type, src_folder)
    gaussian_mix(data_zipped, pitch_type, src_folder)
    dbscan(data_zipped, pitch_type, src_folder)
    k_means(data_zipped, pitch_type, src_folder)
    fuzzy_k_means(data_zipped, pitch_type, src_folder)

    return

def generate_path(algorithm, data, data_name, pitch_type, src_folder):
    return generate_public_paths(algorithm, data, data_name, pitch_type, src_folder)
    if 'std' in data_name:
        std = 'std'
    else:
        std = 'na_std'

    if 'original' in data_name:
        orig_or_group = 'original'
    else:
        orig_or_group = 'grouped'

    dimensions = '_d'
    if data.shape[-1] == 2:
        dimensions = '2d'
    elif data.shape[-1] == 3:
        dimensions = '3d'

    os.makedirs(f'./graphics/{src_folder}/{algorithm}/{pitch_type}/{std}', exist_ok=True)
    os.makedirs(f'./graphics/{src_folder}/{algorithm}/{pitch_type}/{std}/{orig_or_group}', exist_ok=True)
    os.makedirs(f'./graphics/{src_folder}/{algorithm}/{pitch_type}/{std}/{orig_or_group}/{dimensions}', exist_ok=True)
    graphic_path = f'./graphics/{src_folder}/{algorithm}/{pitch_type}/{std}/{orig_or_group}/{dimensions}'

    os.makedirs(f'./labels/{src_folder}/{algorithm}/{pitch_type}/{std}', exist_ok=True)
    os.makedirs(f'./labels/{src_folder}/{algorithm}/{pitch_type}/{std}/{orig_or_group}', exist_ok=True)
    os.makedirs(f'./labels/{src_folder}/{algorithm}/{pitch_type}/{std}/{orig_or_group}/{dimensions}', exist_ok=True)
    label_path = f'./labels/{src_folder}/{algorithm}/{pitch_type}/{std}/{orig_or_group}/{dimensions}'

    return graphic_path, label_path

def generate_public_paths(algorithm, data, data_name, pitch_type, src_folder):
    if 'std' in data_name:
        std = 'std'
    else:
        std = 'na_std'
    if 'original' in data_name:
        orig_or_group = 'original'
    else:
        orig_or_group = 'grouped'
    dimensions = '_d'
    if data.shape[-1] == 2:
        dimensions = '2d'
    elif data.shape[-1] == 3:
        dimensions = '3d'
    os.makedirs(f'../pitch-analyzer/public/graphics/cluster/{src_folder}/{algorithm}/{pitch_type}/{std}', exist_ok=True)
    os.makedirs(f'../pitch-analyzer/public/graphics/cluster/{src_folder}/{algorithm}/{pitch_type}/{std}/{orig_or_group}', exist_ok=True)
    os.makedirs(f'../pitch-analyzer/public/graphics/cluster/{src_folder}/{algorithm}/{pitch_type}/{std}/{orig_or_group}/{dimensions}', exist_ok=True)
    graphic_path = f'../pitch-analyzer/public/graphics/cluster/{src_folder}/{algorithm}/{pitch_type}/{std}/{orig_or_group}/{dimensions}'

    os.makedirs(f'../pitch-analyzer/public/labels/cluster/{src_folder}/{algorithm}/{pitch_type}/{std}', exist_ok=True)
    os.makedirs(f'../pitch-analyzer/public/labels/cluster/{src_folder}/{algorithm}/{pitch_type}/{std}/{orig_or_group}', exist_ok=True)
    os.makedirs(f'../pitch-analyzer/public/labels/cluster/{src_folder}/{algorithm}/{pitch_type}/{std}/{orig_or_group}/{dimensions}', exist_ok=True)
    label_path = f'../pitch-analyzer/public/labels/cluster/{src_folder}/{algorithm}/{pitch_type}/{std}/{orig_or_group}/{dimensions}'

    return graphic_path, label_path


def main():
    os.makedirs('/graphics', exist_ok=True)
    os.makedirs('/labels', exist_ok=True)

    # print(temp)
    # cluster_data(f'stuff_summarized_FF.csv', src_folder = 'pitches_stuff_summarized')
    # cluster_data(f'FF.csv', src_folder = 'pitches_rates_summarized')

    for pitch_file in os.listdir(f'../stats/pitches_stuff_summarized'):
        # print(pitch_file)
        cluster_data(pitch_file, src_folder = 'pitches_stuff_summarized')
    for pitch_file in os.listdir(f'../stats/pitches_rates_summarized'):
        # print(pitch_file)
        cluster_data(pitch_file, src_folder = 'pitches_rates_summarized')

main()