#!/bin/env python3
# coding: utf-8
# Author: Andrey Kutuzov

import gensim
import logging
import multiprocessing
import argparse
from os import path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--corpus", help="Path to a training corpus (can be compressed)", required=True)
    arg("--cores", default=False, help="Limit on the number of cores to use")
    arg("--sg", default=0, type=int, help="Use Skipgram (1) or CBOW (0)")
    arg("--epochs", default=5, type=int, help="Number of training passes over corpora")
    arg("--window", default=5, type=int, help="Size of context window")
    arg("--vocab", default=100000, type=int, help="Max vocabulary size")
    args = parser.parse_args()

    # Setting up logging:
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    # This will be our training corpus to infer word embeddings from.
    # Most probably, a gzipped text file, one doc/sentence per line:
    corpus = args.corpus + ".txt"

    # Iterator over lines of the corpus
    data = gensim.models.word2vec.LineSentence(corpus)

    # How many workers (CPU cores) to use during the training?
    if args.cores:
        # Use the number of cores we are told to use (in a SLURM file, for example):
        cores = int(args.cores)
    else:
        # Use all cores we have access to except one
        cores = (
            multiprocessing.cpu_count() - 1
        )
    logger.info(f"Number of cores to use: {cores}")

    # Setting up training hyperparameters:
    # Use Skipgram (1) or CBOW (0) algorithm?
    skipgram = args.sg
    # Context window size (e.g., 2 words to the right and to the left)
    window = args.window
    # How many words types we want to be considered (sorted by frequency)?
    vocabsize = args.vocab

    vectorsize = 300  # Dimensionality of the resulting word embeddings.

    # For how many epochs to train a model (how many passes over corpus)?
    iterations = args.epochs

    # Start actual training!

    # NB: Subsampling ('sample' parameter) is used to stochastically downplay the influence
    # of very frequent words. If our corpus is already filtered for stop words
    # (functional parts of speech), we do not need subsampling and set it to zero.
    model = gensim.models.Word2Vec(
        data,
        vector_size=vectorsize,
        window=window,
        workers=cores,
        sg=skipgram,
        max_final_vocab=vocabsize,
        epochs=iterations,
        sample=0.001,
    )

    # Saving the resulting model to a file
    output_name = corpus.split("/")[-1]
    #filename = path.basename(output_name).replace(".txt", ".model")
    #logger.info(filename)
    # Save the model without the output vectors (what you most probably want):
    model.wv.save(f'models/{output_name.replace(".txt", ".model")}')

    # model.save(filename)  # If you intend to train the model further
