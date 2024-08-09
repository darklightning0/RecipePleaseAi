import pickle
import plotly.express as px
import umap
from sklearn.cluster import KMeans
import hdbscan
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score

file_path = '/Users/efekapukulak/Desktop/Hobbies/Coding/ML/RecipeAi/recipes_dataset/FoodIngredientsandRecipeDatasetwithImageNameMapping.csv'
df = pd.read_csv(file_path)
with open('gpt_12k_embeddings.pkl', 'rb') as file:
    embeddings = pickle.load(file)
    embeddings = np.array(embeddings)

umap_embeddings = umap.UMAP(n_neighbors=15, n_components=2, metric='cosine').fit_transform(embeddings)


def umap_hdbscan(df, umap_embeddings):
    cluster = hdbscan.HDBSCAN(min_cluster_size=10, metric="euclidean", cluster_selection_method='leaf', min_samples=1).fit_predict(umap_embeddings)

    df_clustered = df.copy()
    df_clustered['Cluster_No'] = cluster
    
    updated_file_path = '/Users/efekapukulak/Desktop/Hobbies/Coding/ML/RecipeAi/recipes_dataset/cluster_dataset_13k.csv'
    df_clustered.to_csv(updated_file_path, index=False)

    df = pd.read_csv(updated_file_path)
    
    cluster_counts = df['Cluster_No'].value_counts()
    
    num_clusters = len(cluster_counts)
    
    top_clusters = cluster_counts.head(10).index

    top_clusters_data = {}
    for cluster_no in top_clusters:
        cluster_recipes = df[df['Cluster_No'] == cluster_no]['Title'].tolist()
        top_clusters_data[cluster_no] = cluster_recipes

    response = {
        "top_clusters": top_clusters_data,
        "num_clusters": num_clusters,
        "recipes_in_cluster_minus_1": cluster_counts.get(-1, 0)
    }
    print(response)

umap_hdbscan(df, umap_embeddings)
