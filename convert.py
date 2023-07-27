#!/bin/env python3
# coding: utf-8
#Author: Andrey Kutuzov
import sys
import gensim
import logging

# This script can be used to convert word embedding models
# from the Gensim format to the standard plain text word2vec format

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)

modelfile = "./models/" + sys.argv[1] + ".model"  # Embedding model in the Gensim native format (*.model)

if modelfile.endswith(".model"):
    model = gensim.models.KeyedVectors.load(modelfile)  # Loading the model
    # model = gensim.models.Word2Vec.load(modelfile) #  If you intend to train the model further

    # Saving the model in Word2vec format (both binary and plain text).
    # If the filename ends in '.gz', Gensim will automatically compress it.
    #model.save_word2vec_format(modelfile.replace(".model", ".vec.gz"), binary=False)
    model.save_word2vec_format(modelfile.replace(".model", ".bin"), binary=True)

elif modelfile.endswith(".vec.gz"):
    model = gensim.models.KeyedVectors.load_word2vec_format(modelfile)  # Loading the model
    model.save_word2vec_format(modelfile.replace(".vec.gz", ".bin"), binary=True)