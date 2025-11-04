#!/usr/bin/env python3
# Multimodal/bayesian/app_bayesian_astar.py
"""
app_bayesian_fft_streamlit_db.py
------------------------------------------------------------------------------
Bayesian A* Graph Demo — Streamlit Visualization for Image Similarity Paths
------------------------------------------------------------------------------
Author: Julia Wen (wendigilane@gmail.com)
Date: 11-03-2025

This Streamlit app constructs a probabilistic image similarity graph and 
computes the most probable paths between images using Monte Carlo sampling 
of Bayesian edge weights.

Key Features:
--------------
1. **Graph Construction**
   - Images are loaded from a dataset directory.
   - Each image is represented as a node.
   - Edges are weighted by perceptual hash similarity and crescent probabilities 
     (from a precomputed JSON file).
   - A k-nearest neighbor (k-NN) graph is built using weighted edges.

2. **Edge Weight Function**
   - Combines visual similarity (via perceptual hash) and crescent probability 
     penalties using tunable global parameters (α, β, and penalty exponent).

3. **Monte Carlo Shortest Path Sampling**
   - Repeatedly perturbs edge weights by Gaussian noise.
   - Computes shortest paths for each sample.
   - Aggregates the most frequently observed (most probable) paths.

4. **Visualization**
   - Displays the constructed graph and highlights top probable paths.
   - Saves graph visualizations and path probabilities to CSV and PNG files.

5. **Streamlit UI**
   - Select source and target images.
   - Adjust number of Monte Carlo samples.
   - View diagnostic statistics and visual path overlays.

Directory Dependencies:
------------------------
- `../graphs/dataset/` — Contains the image dataset.
- `../test/test_files/crescent_probs.json` — Stores per-image crescent probabilities.

Python Version: 3.9+
Dependencies: streamlit, networkx, matplotlib, Pillow, imagehash, numpy, json

------------------------------------------------------------------------------
"""

import os
import random
import networkx as nx
import streamlit as st
import matplotlib.pyplot as plt
import csv
import hashlib
import json
from datetime import datetime
from matplotlib import colormaps
from PIL import Image
import imagehash

# ----------- GLOBAL EDGE WEIGHT PARAMETERS -----------
SIMILARITY_ALPHA = 20.0    # Increase this to make dissimilar images much more expensive
SIMILARITY_BETA = 4.0
SIMILARITY_PENALTY_EXP = 5
K_NEIGHBORS = 5            # Number of nearest neighbors for k-NN graph

DATASET_DIR = os.path.join(os.path.dirname(__file__), "../graphs/dataset")

# ---------------- Utility Functions ----------------

def path_to_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# ---------------- Crescent Probability ----------------

def load_crescent_probs():
    JSON_PATH = os.path.join(os.path.dirname(__file__), "../test/test_files/crescent_probs.json")
    if not os.path.exists(JSON_PATH):
        st.error(f"Crescent probabilities JSON not found: {JSON_PATH}")
        return {}
    with open(JSON_PATH, "r") as f:
        crescent_probs_json = json.load(f)
    # Map by filename only
    crescent_probs = {os.path.basename(k): v for k, v in crescent_probs_json.items()}
    return crescent_probs

# ---------------- Graph Construction ----------------

def similarity_weight(img1_path, img2_path, crescent_prob1, crescent_prob2, 
                      alpha=SIMILARITY_ALPHA, beta=SIMILARITY_BETA, penalty_exp=SIMILARITY_PENALTY_EXP):
    """Weight favors high crescent probability and visually similar images."""
    h1 = imagehash.phash(Image.open(img1_path))
    h2 = imagehash.phash(Image.open(img2_path))
    hash_diff = (h1 - h2) / len(h1.hash)**2  # normalized [0,1]
    sim_score = (1 - hash_diff)**beta
    crescent_factor = (1 - (crescent_prob1 * crescent_prob2))**penalty_exp
    weight = (1 - sim_score) + alpha * crescent_factor
    return max(0.01, weight)  # avoid zero weights

def build_prob_graph(crescent_probs, k=K_NEIGHBORS):  # k = number of neighbors
    G = nx.Graph()
    images = []
    for root, dirs, files in os.walk(DATASET_DIR):
        for f in files:
            if f.endswith((".png", ".jpg", ".jpeg")):
                images.append(os.path.join(root, f))

    if not images:
        st.warning("No images found on disk.")
        return G, {}, [], []

    hash_to_path = {}
    hash_nodes = []
    for img in images:
        h = path_to_hash(img)
        hash_nodes.append(h)
        hash_to_path[h] = img

    # For each image, connect only to its k nearest images (k-NN graph)
    for i, h1 in enumerate(hash_nodes):
        img1 = hash_to_path[h1]
        fname1 = os.path.basename(img1)
        c1 = crescent_probs.get(fname1, 0.0)

        # Compute weights to all others
        weights = []
        for j, h2 in enumerate(hash_nodes):
            if i == j:
                continue
            img2 = hash_to_path[h2]
            fname2 = os.path.basename(img2)
            c2 = crescent_probs.get(fname2, 0.0)
            w = similarity_weight(img1, img2, c1, c2, SIMILARITY_ALPHA, SIMILARITY_BETA, SIMILARITY_PENALTY_EXP)
            weights.append((w, h2))
        # Sort by weight and keep k best neighbors
        weights.sort()
        for w, h2 in weights[:k]:
            if not G.has_edge(h1, h2):
                G.add_edge(h1, h2, mean=w, std=0.05)
    return G, hash_to_path, hash_nodes, images

# ---------------- Monte Carlo Shortest Paths ----------------

def monte_carlo_shortest_path(G, source, target, n_samples=200):
    path_counts = {}
    for _ in range(n_samples):
        sampled_G = nx.Graph()
        for u, v, data in G.edges(data=True):
            w = max(random.gauss(data["mean"], data["std"]), 1e-6)
            sampled_G.add_edge(u, v, weight=w)
        try:
            path = tuple(nx.shortest_path(sampled_G, source, target, weight="weight"))
            path_counts[path] = path_counts.get(path, 0) + 1
        except nx.NetworkXNoPath:
            continue
    return path_counts

# ---------------- Graph Plot Functions ----------------

def plot_graph(G, path=None):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8,6))
    nx.draw(G, pos, with_labels=False, node_size=50, alpha=0.5)
    if path:
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color="red", node_size=80)
        edges_in_path = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=edges_in_path, edge_color="red", width=2)
    plt.title("Graph with Most Probable Path Highlighted")
    st.pyplot(plt, clear_figure=True)

def plot_graph_top_paths(G, top_paths, save_path=None):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10,8))
    nx.draw(G, pos, with_labels=False, node_size=50, alpha=0.3, node_color="gray")
    cmap_colors = colormaps['tab10'].colors
    for idx, (path, _) in enumerate(top_paths):
        color = cmap_colors[idx % len(cmap_colors)]
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color=[color]*len(path), node_size=80)
        edges_in_path = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=edges_in_path, edge_color=[color]*len(edges_in_path), width=2)
    plt.title("Graph with Top 10 Paths Highlighted")
    st.pyplot(plt, clear_figure=True)
    if save_path:
        plt.savefig(save_path)
        st.write(f"Graph saved to: {save_path}")

# ---------------- Streamlit UI ----------------

st.title("Bayesian A* Graph Demo")
st.write("Compute most probable paths through images using Monte Carlo Bayesian edge weights.")

crescent_probs = load_crescent_probs()
G, hash_to_path, hash_nodes, images = build_prob_graph(crescent_probs, k=K_NEIGHBORS)

if len(hash_nodes) < 2:
    st.warning("Not enough images for pathfinding.")
else:
    source = st.selectbox(
        "Select source image",
        images,
        format_func=lambda x: os.path.basename(x),
        key="source_box"
    )

    target = st.selectbox(
        "Select target image (can be same as start)",
        images,
        format_func=lambda x: os.path.basename(x),
        index=len(images)-1,
        key="target_box"
    )

    st.subheader("Start and Target Images")
    col1, col2 = st.columns(2)
    with col1:
        fname_source = os.path.basename(source)
        st.image(source, caption=f"Start Image (crescent prob: {crescent_probs.get(fname_source,0):.4f})", width='stretch')
    with col2:
        fname_target = os.path.basename(target)
        st.image(target, caption=f"Target Image (crescent prob: {crescent_probs.get(fname_target,0):.4f})", width='stretch')

    n_samples = st.slider("Number of Monte Carlo samples", min_value=100, max_value=2000, value=500, step=100)

    if st.button("Compute Paths"):
        source_hash = path_to_hash(source)
        target_hash = path_to_hash(target)

        c1 = crescent_probs.get(fname_source, 0.0)
        c2 = crescent_probs.get(fname_target, 0.0)
        weight = similarity_weight(
            source, target, c1, c2, SIMILARITY_ALPHA, SIMILARITY_BETA, SIMILARITY_PENALTY_EXP
        )
        h1 = imagehash.phash(Image.open(source))
        h2 = imagehash.phash(Image.open(target))
        hash_diff = (h1 - h2) / len(h1.hash)**2
        sim_score = (1 - hash_diff)

        print("\n[DEBUG] Selected Images Debug:")
        print(f"Source: {fname_source}, crescent_prob={c1:.4f}")
        print(f"Target: {fname_target}, crescent_prob={c2:.4f}")
        print(f"Similarity score: {sim_score:.4f}")
        print(f"Computed edge weight: {weight:.4f}")

        # Show direct edge weight from the graph for comparison
        edge_data = G.get_edge_data(source_hash, target_hash)
        print(f"[DEBUG] Direct edge weight in graph: {edge_data}")

        # Try all 2-step paths:
        for node in hash_nodes:
            if node != source_hash and node != target_hash:
                w1 = G.get_edge_data(source_hash, node)
                w2 = G.get_edge_data(node, target_hash)
                if w1 and w2:
                    total = w1["mean"] + w2["mean"]
                    print(f"2-step path via {os.path.basename(hash_to_path[node])}: {w1['mean']:.2f} + {w2['mean']:.2f} = {total:.2f}")

        results = monte_carlo_shortest_path(G, source_hash, target_hash, n_samples=n_samples)
        total = sum(results.values())
        st.subheader("Most Probable Paths")

        st.subheader("Path Length Diagnostics")
        path_lengths = [len(p) for p in results.keys()]
        if path_lengths:
            mean_len = sum(path_lengths) / len(path_lengths)
            std_len = (sum((x - mean_len) ** 2 for x in path_lengths) / len(path_lengths)) ** 0.5
            st.write(f"Mean length: {mean_len:.2f}")
            st.write(f"Std. dev.: {std_len:.2f}")
            st.write(f"Min length: {min(path_lengths)}")
            st.write(f"Max length: {max(path_lengths)}")

        top_paths = sorted(results.items(), key=lambda x: -x[1])[:10]
        for path, count in top_paths:
            st.write(f"{[os.path.basename(hash_to_path.get(p, p)) for p in path]}: {count / total:.2%}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.getcwd(), "test_output")
        os.makedirs(output_dir, exist_ok=True)

        top_path = top_paths[0][0]
        plot_graph(G, top_path)

        colored_graph_path = os.path.join(output_dir, f"graph_top10_paths_{timestamp}.png")
        plot_graph_top_paths(G, top_paths, save_path=colored_graph_path)

        csv_path = os.path.join(output_dir, f"top_paths_{timestamp}.csv")
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Path (filenames)", "Probability"])
            for path, count in top_paths:
                filenames = [os.path.basename(hash_to_path.get(p, p)) for p in path]
                probability = count / total
                writer.writerow([" -> ".join(filenames), f"{probability:.4f}"])
        st.write(f"Top 10 paths saved to: {csv_path}")
