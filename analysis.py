#!/bin/env python3
# coding: utf-8
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
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation, SpectralClustering, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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
    #languages = ["Norwegian","German", "English" ,"Spanish", "French", "Russian" , "Arabic", "Hebrew", "Somali", "Wolaytta", "Tachelhit"]
    languages_og = ["Wolof", "Xhosa", "Zulu", "Ewe", "Finnish", "Estonian", "Hungarian", "Spanish", "German", "Lithuanian", "Russian", "Hebrew", "Arabic", "Tachelhit", "Somali"]
    languages = []
    for language in languages_og:
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
        random.shuffle(data_into_list)
        for line in data_into_list[:50]:
            words = line.split(" ")
            words = [w.translate(str.maketrans('', '', string.punctuation)) for w in words]
            sentence_vector = np.mean([model[word] for word in words if word in model], axis=0)
            if sentence_vector.size == 500:
                lang_averages.append(sentence_vector)
                languages.append(language)
        #language_average = np.mean(np.array(sentences), axis=0)
        #lang_averages.append(language_average)

    n_clusters = 4
    df = pd.DataFrame(languages)
    df["embedding"] = lang_averages
    matrix = np.vstack(df.embedding.values)
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(matrix)
    pca = PCA(n_components=3)
    results = pca.fit_transform(scaled_matrix)
    results = pd.DataFrame(results)
    results['x'] =results.iloc[0:, 0]
    results['y'] =results.iloc[0:, 1]
    results['z'] =results.iloc[0:, 2]
    tmp = pd.merge(df, results, left_index=True, right_index=True)
    tmp['language'] =tmp.iloc[0:, 0]
    df =tmp[['language', 'embedding', 'x', 'y', 'z']]

    fig = plt.figure(figsize=(14,9))
    ax = fig.add_subplot(111, projection='3d')
    for category, color in enumerate(["green", "red", "blue", "pink"]):
        pass
        #xs = np.array(df["x"])[df.cluster == category]
        #ys = np.array(df["y"])[df.cluster == category]
        #ax.scatter(xs, ys, color=color, alpha=0.3)
    for i, txt in enumerate(df["language"]):
        #ax.annotate(txt, (df["x"][i], df["y"][i]))
        if txt in ["Wolof", "Xhosa", "Zulu", "Ewe"]:
            ax.plot(df["x"][i], df["y"][i], df["z"][i], color="green", marker='o', linestyle='', ms=3)
        if txt in ["Finnish", "Estonian", "Hungarian"]:
            ax.plot(df["x"][i], df["y"][i], df["z"][i], color="red", marker='o', linestyle='', ms=3)
        if txt in ["Spanish", "German", "Lithuanian", "Russian"]:
            ax.plot(df["x"][i], df["y"][i], df["z"][i], color="blue", marker='o', linestyle='', ms=3)
        if txt in ["Hebrew", "Arabic", "Tachelhit", "Somali"]:
            ax.plot(df["x"][i], df["y"][i], df["z"][i], color="orange", marker='o', linestyle='', ms=3)

    import matplotlib.patches as mpatches
    green_patch = mpatches.Patch(color='green', label="Niger-Congo")
    red_patch = mpatches.Patch(color='red', label="Uralic")
    blue_patch = mpatches.Patch(color='blue', label="Indo-European")
    orange_patch = mpatches.Patch(color='orange', label="Afro-Asiatic")

    plt.legend(handles=[green_patch, red_patch, blue_patch, orange_patch])

    plt.show()