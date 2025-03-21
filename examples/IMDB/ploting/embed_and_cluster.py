from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd

from sklearn.cluster import KMeans
import pdb

# Load model and tokenizer
model_name = "EleutherAI/gpt-j-6b" #"gpt2"  # Example using GPT-2
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

reviews = pd.read_csv('/mnt/shared-scratch/Anon1/auth1/rlhf/trl/examples/IMDB/ploting/critical_neg_texts.csv')
reviews_list = reviews['review'].tolist()

# get embeddings 
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    return hidden_states[-1][:, 0, :].squeeze()

embeddings = [get_embedding(review) for review in reviews_list]

# cluster embeddings 
# Assuming embeddings is a list of embeddings generated as shown above
embeddings_array = torch.stack(embeddings).cpu().numpy()

# Perform K-Means clustering
#pdb.set_trace()
n_clusters = 5  # Example number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(embeddings_array)

# Getting the cluster labels
labels = kmeans.labels_

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Step 1: Dimensionality Reduction
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
reduced_embeddings = pca.fit_transform(embeddings_array)

# Step 2: Plot Clusters
plt.figure(figsize=(10, 8))
for i in range(n_clusters):
    plt.scatter(reduced_embeddings[labels == i, 0], reduced_embeddings[labels == i, 1], label=f'Cluster {i}')
plt.title("Cluster Visualization")
plt.savefig("Cluster_Visualization", bbox_inches="tight")
plt.legend()
plt.show()

# Step 3: Output Reviews from Each Cluster
# Assuming 'your_reviews_list' is the list of reviews corresponding to the embeddings
reviews_df = pd.DataFrame({
    'Review': reviews_list,
    'Cluster': labels
})

for i in range(n_clusters):
    print(f"\nReviews from Cluster {i}:\n")
    sample_reviews = reviews_df[reviews_df['Cluster'] == i].sample(3, random_state=42)  # Sample 3 reviews
    for index, row in sample_reviews.iterrows():
        print(f"- {row['Review']}\n")