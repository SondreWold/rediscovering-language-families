#!/bin/env python3
# coding: utf-8

# To run this on Fox, load this module:
# nlpl-gensim/4.3.1-foss-2021a-Python-3.9.5

import sys
import gensim
import logging
import json
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import zipfile
import string
from os import path
import torch
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation, SpectralClustering
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Simple toy script to get an idea of what one can do with (static) word embedding models using Gensim
# Models can be found at http://vectors.nlpl.eu/explore/embeddings/models/,
# or http://vectors.nlpl.eu/repository/,
# or in the /fp/projects01/ec30/models/static/  directory on Fox
# (for example, /fp/projects01/ec30/models/static/223/)


def load_embedding(modelfile):
    # Detect the model format by its extension:
    # Binary word2vec format:
    if modelfile.endswith(".bin.gz") or modelfile.endswith(".bin"):
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            modelfile, binary=True, unicode_errors="replace"
        )
    # Text word2vec format:
    elif (
        modelfile.endswith(".txt.gz")
        or modelfile.endswith(".txt")
        or modelfile.endswith(".vec.gz")
        or modelfile.endswith(".vec")
    ):
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            modelfile, binary=False, unicode_errors="replace"
        )
    # ZIP archive from the NLPL vector repository:
    elif modelfile.endswith(".zip"):
        with zipfile.ZipFile(modelfile, "r") as archive:
            stream = archive.open(
                "model.bin"  # or model.txt, if you want to look at the model
            )
            emb_model = gensim.models.KeyedVectors.load_word2vec_format(
                stream, binary=True, unicode_errors="replace"
            )
    else:  # Native Gensim format?
        emb_model = gensim.models.KeyedVectors.load(modelfile)
        #  If you intend to train the model further:
        # emb_model = gensim.models.Word2Vec.load(embeddings_file)
    return emb_model


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    lang_averages = []
    #"Finnish", "Hungarian", "Estonian"
    languages = ["Norwegian","German", "Danish" ,"Spanish", "French", "Russian" , "Arabic", "Hebrew", "Somali", "Wolaytta", "Tachelhit"]
    for language in languages:
        embeddings_file = path.join(
            "./models/", f"{language}.bin"
        )  # Directory containing word embeddings
        text_file = f"parsed_corpora/{language}.txt"

        data_into_list = None
        with open(text_file, "r") as f:
            data = f.read()
            data_into_list = data.split("\n")

        model = load_embedding(embeddings_file)
        sentences = []
        for line in data_into_list:
            words = line.split(" ")
            words = [w.translate(str.maketrans('', '', string.punctuation)) for w in words]
            sentence_vector = np.mean([model[word] for word in words if word in model], axis=0)
            if sentence_vector.size == 300:
                sentences.append(sentence_vector)
        language_average = np.mean(np.array(sentences), axis=0)
        lang_averages.append(language_average)

    n_clusters = 2
    df = pd.DataFrame(languages)
    df["embedding"] = lang_averages
    #clusterer = clusterer(n_clusters=n_clusters, init="k-means++", random_state=42)
    clusterer = SpectralClustering(n_clusters=n_clusters)
    matrix = np.vstack(df.embedding.values)
    clusterer.fit(matrix)
    labels = clusterer.labels_
    df["cluster"] = labels
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(matrix)
    pca = PCA(n_components=3)
    results = pca.fit_transform(scaled_matrix)
    results = pd.DataFrame(results)
    results['x'] =results.iloc[0:, 0]
    results['y'] =results.iloc[0:, 1]
    tmp = pd.merge(df, results, left_index=True, right_index=True)
    tmp['language'] =tmp.iloc[0:, 0]
    df =tmp[['language', 'embedding', 'x', 'y', 'cluster']]

    fig, ax = plt.subplots()
    for category, color in enumerate(["green", "red"]):
        xs = np.array(df["x"])[df.cluster == category]
        ys = np.array(df["y"])[df.cluster == category]
        ax.scatter(xs, ys, color=color, alpha=0.3)
    for i, txt in enumerate(df["language"]):
        ax.annotate(txt, (df["x"][i], df["y"][i]))
    plt.show()