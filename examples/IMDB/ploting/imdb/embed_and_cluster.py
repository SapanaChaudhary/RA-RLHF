from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd

from sklearn.cluster import KMeans
import pdb

# Load model and tokenizer
model_name = "EleutherAI/gpt-j-6b" #"gpt2"  # Example using GPT-2
tokenizer = AutoTokenizer.from_pretrained(model_name, add_special_tokens=False, unk_token=None, keep_accents=False)
model = AutoModelForCausalLM.from_pretrained(model_name)

reviews_1 = pd.read_csv('/mnt/shared-scratch/Anon1/auth1/rlhf/trl/examples/IMDB/ploting/imdb/critical_neg_texts.csv')
reviews_list = reviews_1['review'].tolist()[:150]

reviews_2 = pd.read_csv('/mnt/shared-scratch/Anon1/auth1/rlhf/trl/examples/IMDB/ploting/imdb/critical_pos_texts.csv')
reviews_list.extend(reviews_2['review'].tolist())

# get embeddings 
def get_embedding(text):
    print('here')
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
n_clusters = 8  # Example number of clusters
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
fig, axs2 = plt.subplots(figsize=(6,6))
for i in range(n_clusters):
    plt.scatter(reduced_embeddings[labels == i, 0], reduced_embeddings[labels == i, 1], s=50, label=f'Cluster {i}')
axs2.legend()
axs2.set_ylabel(r'x', fontsize=16) # Setting label size
axs2.set_xlabel(r'y', fontsize=16)
axs2.tick_params(axis='both', which='major', labelsize=14) # Adjusting tick size
axs2.set_box_aspect(1)
axs2.grid(True) # Additional aesthetics
plt.show()
plt.savefig("Cluster_Visualization.pdf", bbox_inches="tight")

# Step 3: Output Reviews from Each Cluster
# Assuming 'your_reviews_list' is the list of reviews corresponding to the embeddings
reviews_df = pd.DataFrame({
    'Review': reviews_list,
    'Cluster': labels
})

for i in range(n_clusters):
    print(f"\nReviews from Cluster {i}:\n")
    #sample_reviews = reviews_df[reviews_df['Cluster'] == i].sample(3, random_state=42)  # Sample 3 reviews
    sample_reviews = reviews_df[reviews_df['Cluster'] == i].sample(3, replace=True, random_state=42)
    for index, row in sample_reviews.iterrows():
        print(f"- {row['Review']}\n")