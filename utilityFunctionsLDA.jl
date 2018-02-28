include("definitions.jl")
using Distributions

function logOfSum(log_x)
    max_log_x = max(log_x)
    max_log_x + log(sum(exp(log_x - max_log_x)))
end

function sampleFromDiscrete(probs)
#Draw a sample from a discrete distribution
    temp = rand();
    total = 0.0;
    for i = 1:length(probs)
        total = total + probs[i];
        if temp < total
            return i;
        end
    end
    show(probs)
    show(sum(probs))
    error("failed to sample from discrete, should never happen!\n");
end

function sampleFromDirichlet(dirParams)
#Draw a sample from a Dirichlet distribution
    #sample = gamrnd(dirParams,ones(size(dirParams)));  #matlab
    sample = zeros(length(dirParams))
    for i = 1:length(sample)
        sample[i] = rand(Gamma(dirParams[i]));
    end
    
    sample = sample ./ sum(sample);
    if sum(isnan(sample)) > 0
        sample
        dirParams
        error("sample contained a NaN!");
    end
    sample
end

function generateSyntheticLDATopics(beta, numTopics, dictionarySize)
#Generate completely synthetic topics, with numbers as words.
    #dict = containers.Map(1:dictionarySize, regexp(num2str(1:dictionarySize), '\s+', 'split')); TODO how to do this in Julia!
    phi = zeros(dictionarySize, numTopics)
    prior = ones(dictionarySize) .* beta
    for i = 1:numTopics
       phi[:,i] = sampleFromDirichlet(prior)
    end
    #(phi, dict)
    phi
end

function generateLDA(V, numDocuments, documentLength, alphaVec)
#generate some documents from LDA
    documents = Array(Document, numDocuments)
    for i = 1:numDocuments
       documents[i] = Document(zeros(Int32,documentLength));
       theta_i = sampleFromDirichlet(alphaVec);
       for j = 1:documentLength
           topic = sampleFromDiscrete(theta_i);
           documents[i].wordVector[j] = sampleFromDiscrete(V[:,topic]);
       end
    end
    documents
end

function computeTopics(VBparams)
    #takes the current parameters, and computes the corresponding estimated topics
    #phi is numWords x numTopics
    
    #phi = bsxfun(+, VBparams.wordTopicCounts, VBparams.eta);
    #phi = bsxfun(./, phi, sum(phi,1));
    
    phi = VBparams.wordTopicCounts .+ VBparams.eta;
    phi ./= sum(phi,1);    
    phi
end

#convert the non-sparse data structure for documents to the sparse data structure (token index, token count pairs)
function wordVectorToSparseCounts(documents::Array{Document,1})
    sparse_documents = Array{SparseDocument}(length(documents))

    for i = 1:length(documents)
        tokensFound = Dict{Int64, Int64}()

        numTokensFound = 0
        for j = 1:length(documents[i].wordVector)
            word = documents[i].wordVector[j];
            if !haskey(tokensFound,word)
                numTokensFound += 1
                tokensFound[word] = numTokensFound;
            end
        end

        sparse_documents[i] = SparseDocument(Int64[], zeros(Int64, numTokensFound), zeros(Int64, numTokensFound), length(documents[i].wordVector)) 

        for j = 1:length(documents[i].wordVector)
            word = documents[i].wordVector[j];
            token = tokensFound[word]
            sparse_documents[i].tokenIndex[token] = word
            sparse_documents[i].tokenCount[token] += 1;
        end
        sparse_documents[i].wordVector = copy(documents[i].wordVector)
    end

    sparse_documents 
end