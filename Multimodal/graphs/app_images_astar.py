# app_images_astar.py
# -------------------------------------------
# * Author: Julia Wen wendigilane@gmail.com
# * Date: 2025-09-16
# Streamlit Demo: A* Search on Image Knowledge Graph
# Using dataset as an example (dataset/crescent, dataset/no_crescent)
# Shows the optimal path between images based on feature similarity
#
# To run:
# 1. Ensure your dataset structure:
#    <project-folder>/
#    ├─ dataset/
#    │  ├─ crescent/
#    │  └─ no_crescent/
#    ├─ app_images_astar.py
# 2. Install dependencies from requirements.txt:
#    pip install -r requirements.txt
# 3. Run the demo:
#    streamlit run app_images_astar.py
# -------------------------------------------

import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import os
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

# -------------------------------
# 1. Load images from dataset folder
# -------------------------------
crescent_dir = "dataset/crescent"
no_crescent_dir = "dataset/no_crescent"

image_paths = []
for d in [crescent_dir, no_crescent_dir]:
    if os.path.exists(d):
        for f in os.listdir(d):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(d, f))
    else:
        st.error(f"Directory not found: {d}")
        st.stop()

image_names = [os.path.basename(p) for p in image_paths]

# -------------------------------
# 2. Extract image embeddings using pretrained ResNet
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model = torch.nn.Sequential(*list(model.children())[:-1])  # remove last layer
model.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

embeddings = []
with torch.no_grad():
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(device)
        emb = model(img_t).squeeze().cpu().numpy()
        embeddings.append(emb)
embeddings = np.array(embeddings)

# -------------------------------
# 3. Build graph based on feature similarity
# -------------------------------
G = nx.Graph()
for i, name in enumerate(image_names):
    G.add_node(name, path=image_paths[i])

num_nodes = len(image_names)
for i in range(num_nodes):
    for j in range(i+1, num_nodes):
        dist = cosine_distances([embeddings[i]], [embeddings[j]])[0][0]
        G.add_edge(image_names[i], image_names[j], weight=dist)

# -------------------------------
# 4. Heuristic function for A*
# -------------------------------
def heuristic(u, v):
    idx_u = image_names.index(u)
    idx_v = image_names.index(v)
    return cosine_distances([embeddings[idx_u]], [embeddings[idx_v]])[0][0]

# -------------------------------
# 5. Streamlit UI
# -------------------------------
st.title("A* Knowledge Graph Demo (Images)")

start_node = st.selectbox("Select Start Image", image_names)
goal_node = st.selectbox("Select Goal Image", image_names, index=1)

if start_node == goal_node:
    st.warning("Start and goal images must be different.")
else:
    # -------------------------------
    # 6. Compute A* path
    # -------------------------------
    path = nx.astar_path(G, start_node, goal_node, heuristic=heuristic, weight='weight')
    path_length = nx.astar_path_length(G, start_node, goal_node, heuristic=heuristic, weight='weight')

    st.success(f"Optimal path from {start_node} to {goal_node}:")
    st.info(f"Total path cost: {path_length:.4f}")

    # -------------------------------
    # 7. Display images along the path
    # -------------------------------
    st.subheader("Images along the path:")
    for node in path:
        st.image(G.nodes[node]['path'], caption=node, width=200)

    # -------------------------------
    # 8. Optional: visualize the graph
    # -------------------------------
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10,8))
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_edges(G, pos, width=1, alpha=0.3)
    nx.draw_networkx_labels(G, pos, font_size=8)

    path_edges = list(zip(path, path[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=3, edge_color='red')
    nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='orange', node_size=500)

    plt.title(f"A* Path from {start_node} to {goal_node}", fontsize=12)
    plt.axis('off')
    st.pyplot(plt)

