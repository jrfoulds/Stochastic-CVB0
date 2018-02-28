# Stochastic-CVB0
Code implementing Stochastic CVB0 (SCVB0), from: [J. R. Foulds, L. Boyles, C. DuBois, P. Smyth and M. Welling. Stochastic collapsed variational Bayesian inference for latent Dirichlet allocation. Proceedings of the 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2013](https://dl.acm.org/citation.cfm?id=2487697). 

## Prerequisites

* [Julia programming language](https://julialang.org/) (tested on v0.6.2)
* The Distributions Julia package, which can be added using the package manager

The code should work on any platform.

## Data format

Routines are provided to read in corpora in two formats: a simple format where each token is individually encoded, and a sparse format.  Both formats involve a text file which has one line for each document in the corpus.  One-based indexing is always used, since Julia indexes arrays this way.  In the simple, non-sparse format (readData.jl), each word in the document is represented by a word index, and these are space separated.  The words and documents should ideally be shuffled, to improve the performance of the stochastic algorithm.

In the sparse format (readSparseData.jl), there is an entry for each distinct word type appearing in a document.  These are encoded by pairs of token index and token count.  All values are space separated, and a newline once again signifies the end of a document.  For reporting the top words in a corpus, a dictionary file is also expected, containing one line per word type in plain text.

The [NIPS corpus, due to Sam Roweis](https://cs.nyu.edu/~roweis/data.html), is encoded in both formats in the data folder (NIPS.txt and NIPSsparse.txt), along with a dictionary file (NIPSdict.txt).

## Running the code

Try out runNIPS.jl to run the algorithm on the NIPS corpus.  A flag switches between the two data formats.  This file can be modified to load your particular dataset.

## Author

* [**James Foulds**](http://jfoulds.informationsystems.umbc.edu/)

## License
Licensed under the Apache License, Version 2.0 (the "License"). You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 .
