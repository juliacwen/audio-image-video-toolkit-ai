#!/usr/bin/env python3
# Multimodal/bayesian/app_bayesian_astar_db.py
"""
------------------------------------------------------------------------------
Bayesian A* Graph Demo — Streamlit Visualization for Image Similarity Paths
------------------------------------------------------------------------------
Author: Julia Wen (wendigilane@gmail.com)
Date: 11-07-2025

This Streamlit app constructs a probabilistic image similarity graph and 
computes the most probable paths between images using Monte Carlo sampling 
of Bayesian edge weights.

Key Features:
--------------
1. **Graph Construction**
   - Images are loaded from a dataset  directory.
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
   - Saves both graph PNGs and top paths into the database.

5. **Streamlit UI**
   - Select source and target images.
   - Adjust number of Monte Carlo samples.
   - View diagnostic statistics and visual path overlays.

Directory Dependencies:
------------------------
- `../graphs/dataset/` — Contains the image dataset.
- `../test/test_files/crescent_probs.json` — Stores per-image crescent probabilities.
- `../db/image_graph_db.py` — Database module for saving runs.
------------------------------------------------------------------------------
"""

import os
import sys
import random
import hashlib
import json
import csv
from io import BytesIO
from datetime import datetime

# ensure project root is importable so `db` package resolves
PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from PIL import Image
import imagehash

from db.image_graph_db import init_image_graph_db, save_images, save_edges, save_run

# ---------- CONFIG ----------
SIMILARITY_ALPHA = 20.0
SIMILARITY_BETA = 4.0
SIMILARITY_PENALTY_EXP = 5
K_NEIGHBORS = 5
MONTE_CARLO_EDGE_STD = 0.05
DEFAULT_MC_SAMPLES = 500

BASE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.normpath(os.path.join(BASE_DIR, "../graphs/dataset"))
CRESCENT_JSON_PATH = os.path.normpath(os.path.join(BASE_DIR, "../test/test_files/crescent_probs.json"))

# Ensure DB exists
init_image_graph_db()


# ---------- utilities ----------
def path_to_hash(path):
    """Stable MD5 for file contents (used as node id)."""
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return hashlib.md5(path.encode("utf-8")).hexdigest()


def load_crescent_probs():
    """Load crescent probability mapping (basename -> prob)."""
    if not os.path.exists(CRESCENT_JSON_PATH):
        return {}
    with open(CRESCENT_JSON_PATH, "r") as f:
        d = json.load(f)
    return {os.path.basename(k): float(v) for k, v in d.items()}


def similarity_weight(img1_path, img2_path, crescent_prob1, crescent_prob2,
                      alpha=SIMILARITY_ALPHA, beta=SIMILARITY_BETA, penalty_exp=SIMILARITY_PENALTY_EXP):
    """
    Compute an edge weight: lower for similar images & lower crescent-penalty.
    Returns positive float (higher = more costly).
    """
    try:
        h1 = imagehash.phash(Image.open(img1_path))
        h2 = imagehash.phash(Image.open(img2_path))
        hash_diff = (h1 - h2) / (len(h1.hash) ** 2)
    except Exception:
        hash_diff = 1.0 if os.path.basename(img1_path) != os.path.basename(img2_path) else 0.0
    sim_score = max(0.0, (1.0 - hash_diff)) ** beta
    crescent_factor = (1.0 - (crescent_prob1 * crescent_prob2)) ** penalty_exp
    weight = (1.0 - sim_score) + alpha * crescent_factor
    return max(1e-6, float(weight))


def build_prob_graph(crescent_probs, k=K_NEIGHBORS):
    """
    Build undirected k-NN graph. Node IDs are md5(file bytes).
    Edges store 'mean' and 'std' for Monte Carlo sampling.
    """
    G = nx.Graph()
    images = []
    for root, _, files in os.walk(DATASET_DIR):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                images.append(os.path.join(root, f))
    if not images:
        return G, {}, [], []
    hash_to_path = {}
    hash_nodes = []
    for img in images:
        h = path_to_hash(img)
        hash_nodes.append(h)
        hash_to_path[h] = img
    # compute all pair weights
    for i, h1 in enumerate(hash_nodes):
        img1 = hash_to_path[h1]
        c1 = crescent_probs.get(os.path.basename(img1), 0.0)
        weights = []
        for j, h2 in enumerate(hash_nodes):
            if i == j:
                continue
            img2 = hash_to_path[h2]
            c2 = crescent_probs.get(os.path.basename(img2), 0.0)
            w = similarity_weight(img1, img2, c1, c2)
            weights.append((w, h2))
        weights.sort(key=lambda x: x[0])
        for w, h2 in weights[:k]:
            if not G.has_edge(h1, h2):
                G.add_edge(h1, h2, mean=float(w), std=float(MONTE_CARLO_EDGE_STD))
    return G, hash_to_path, hash_nodes, images


def monte_carlo_shortest_path(G, source, target, n_samples=200):
    """
    Sample edge weights n_samples times and count shortest paths occurrences.
    Returns mapping {path_tuple: count}
    """
    path_counts = {}
    if source not in G or target not in G:
        return path_counts
    for _ in range(n_samples):
        sampled = nx.Graph()
        for u, v, data in G.edges(data=True):
            mean = float(data.get("mean", 1.0))
            std = float(data.get("std", MONTE_CARLO_EDGE_STD))
            w = max(random.gauss(mean, std), 1e-9)
            sampled.add_edge(u, v, weight=w)
        try:
            path = tuple(nx.shortest_path(sampled, source, target, weight="weight"))
            path_counts[path] = path_counts.get(path, 0) + 1
        except nx.NetworkXNoPath:
            continue
    return path_counts

def plot_graph_bytes(G, path=None, top_paths=None, figsize=(8, 6), title=None):
    """
    Render graph to PNG bytes. Title can be provided to distinguish the graph type.
    """
    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=figsize)
    nx.draw(G, pos, with_labels=False, node_size=30, alpha=0.4, ax=ax)

    if path:
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color="red", node_size=90, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=list(zip(path, path[1:])), edge_color="red", width=2, ax=ax)

    if top_paths:
        cmap_colors = colormaps['tab10'].colors
        for idx, (p, _) in enumerate(top_paths):
            color = cmap_colors[idx % len(cmap_colors)]
            nx.draw_networkx_nodes(G, pos, nodelist=p, node_color=[color]*len(p), node_size=70, ax=ax)
            nx.draw_networkx_edges(G, pos, edgelist=list(zip(p, p[1:])), edge_color=[color]*len(p), width=2, ax=ax)

    # ✅ Correct title applied
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Graph")  # fallback

    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()



# ---------- Streamlit UI ----------
st.set_page_config(layout="wide", page_title="Bayesian A* Graph Demo")
st.markdown("## Bayesian A* Graph — Monte Carlo Bayesian Edge Sampling")

crescent_probs = load_crescent_probs()
G, hash_to_path, hash_nodes, images = build_prob_graph(crescent_probs, k=K_NEIGHBORS)

if hash_nodes:
    save_images(hash_to_path, crescent_probs)
    save_edges(G)

if not images:
    st.warning("No images found under: " + DATASET_DIR)
    st.stop()

if len(hash_nodes) < 2:
    st.warning("Not enough images for pathfinding.")

source = st.selectbox("Source image", images, format_func=lambda x: os.path.basename(x))
target = st.selectbox("Target image", images, format_func=lambda x: os.path.basename(x), index=len(images)-1)

col1, col2 = st.columns(2)
with col1:
    st.image(source, caption=f"Source: {os.path.basename(source)}", width=300)
with col2:
    st.image(target, caption=f"Target: {os.path.basename(target)}", width=300)

n_samples = st.slider("Monte Carlo samples", min_value=100, max_value=2000, value=DEFAULT_MC_SAMPLES, step=100)

if st.button("Compute Paths"):
    source_hash = path_to_hash(source)
    target_hash = path_to_hash(target)
    results = monte_carlo_shortest_path(G, source_hash, target_hash, n_samples)
    total = sum(results.values()) if results else 0
    if not results:
        st.write("No paths found between selected images.")
    else:
#        st.subheader("Most probable paths (top 10)")
        top_paths = sorted(results.items(), key=lambda x: -x[1])[:10]
        normalized = [(list(p), cnt/total) for p, cnt in top_paths]

        # Save CSV & PNGs to disk
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(os.getcwd(), "test_output")
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"top_paths_{timestamp}.csv")
        png_most_path = os.path.join(out_dir, f"graph_most_probable_{timestamp}.png")
        png_top10_path = os.path.join(out_dir, f"graph_top10_paths_{timestamp}.png")

        # Write CSV
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Path (filenames)", "Probability"])
            for p, prob in normalized:
                fnames = [os.path.basename(hash_to_path.get(h, h)) for h in p]
                writer.writerow([" -> ".join(fnames), f"{prob:.6f}"])

        # Plot PNGs
        most_prob_path = normalized[0][0] if normalized else None
        png_most = plot_graph_bytes(G, path=most_prob_path,title="Graph with Most Probable Path Highlighted")
        png_top10 = plot_graph_bytes(G, top_paths=normalized, title="Graph with Top 10 Paths Highlighted")

        with open(png_most_path, "wb") as f:
            f.write(png_most)
        with open(png_top10_path, "wb") as f:
            f.write(png_top10)

        # Save run to DB with sample_count
        run_name = f"image_graph_run_{timestamp}"
        db_top_paths = [([os.path.basename(hash_to_path.get(h, h)) for h in p], prob) for p, prob in normalized]
        save_run(
            run_name=run_name,
            source_image=os.path.basename(source),
            target_image=os.path.basename(target),
            graph_most_probable_bytes=png_most,
            graph_top10_bytes=png_top10,
            top_paths=db_top_paths,
            sample_count=n_samples
        )

        # Show images then logs
        st.image(png_most, use_container_width=True)
        st.image(png_top10, use_container_width=True)
        st.write(f"Run saved to DB as: {run_name}")
        st.write(f"CSV saved: {csv_path}")
        st.write(f"Graph PNGs saved: {png_most_path}, {png_top10_path}")
