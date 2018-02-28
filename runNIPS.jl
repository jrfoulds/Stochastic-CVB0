include("readData.jl")
include("readSparseData.jl")
include("saveSparseData.jl")
include("stochasticCVB0.jl")

#This file shows a demo of the SCVB0 algorithm on the NIPS data.

numWords = 13649
numTopics = 50
minibatchSize = 20
numDocumentIterations = 50000
time_limit = 120.0
burnIn = 1
alpha = 0.1
eta = 0.01
tau = 1000.0
kappa = 0.9
tau2= 1.0
kappa2 = 0.9
scale = 100.0
saveParams = true
saveCount = 100

dataFilename = "data/NIPS.txt";                 #one line per document, space-separated one-based dictionary indices for each consecutive word in the document.
sparseDataFilename = "data/NIPSsparse.txt";     #sparse file format, obtained from the dense data format via saveSparseData("data/NIPSsparse.txt", documents);
dictionaryFilename = "data/NIPSdict.txt";       #one line per dictionary word

readSparseDataset = true; #This script demonstrates reading either the non-sparse file format or the sparse file format, selected here

if readSparseDataset
    documents, dictionary = readSparseData(sparseDataFilename, dictionaryFilename);
else
    documents, dictionary = readData(dataFilename, dictionaryFilename)
    println("Converting to sparse format...")
    documents = wordVectorToSparseCounts(documents);
    println("Done.")
end

tic();
VBparams,saved_iters, saved_time, saved_topics, saved_alphas  = stochasticCVB0(documents,numWords,numTopics,
                              numDocumentIterations,time_limit,burnIn,minibatchSize,alpha,eta,
                              dictionary,tau,kappa,tau2,kappa2,scale,saveParams,saveCount);
t = toc();

phi = computeTopics(VBparams); #compute the normalized topic distributions