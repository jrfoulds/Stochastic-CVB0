include("utilityFunctionsLDA.jl")
include("update_functions.jl")
include("getImportantWordsInTopic.jl")

function stochasticCVB0(documents::Array{SparseDocument,1},
                                                    numWords::Int,
                                                    numTopics::Int,
                                                    numDocumentIterations::Int,
                                                    time_limit::Float64,
                                                    burnInPerDoc::Int,
                                                    minibatchSize::Int,
                                                    alpha::Float64,
                                                    eta::Float64,
                                                    dict,
                                                    tau::Float64,
                                                    kappa::Float64,
                                                    tau2::Float64,
                                                    kappa2::Float64,
                                                    scale::Float64,
                                                    saveParams,
                                                    saveCount,
                                                    initParams...)
#Stochastic Collapsed Variational Bayes for LDA
#@author Jimmy Foulds, Levi Boyles
    savepoints = exp.(linspace(0,log(time_limit),saveCount))
    saveInd = 1

    debugTopics = true;
    meanchangeThreshold = 0.01;
    alpha = ones(1,numTopics) .* alpha;
    eta = ones(numWords,1).*eta;

    totalWordsInCorpus = 0;
    for i = 1:length(documents)
        totalWordsInCorpus = totalWordsInCorpus + length(documents[i].wordVector);
    end
 
    if length(initParams) == 0
        VBparams = initialize(documents, alpha, eta, numWords, totalWordsInCorpus);
    else
        println("Initializing to given params!");
        VBparams = initParams[1];
    end
   
        
    wordTopicCounts_hat = zeros(numWords,numTopics);
    topicCounts_hat = zeros(1, numTopics);

    miniBatchesPerCorpus = length(documents) ./ minibatchSize;
    VBparams.iter = 1;
    stepSize = scale ./tau^kappa;
   
    saved_topics = Array{Array{Float64, 2}}(0)
    saved_alphas = Array{Array{Float64, 2}}(0)
    saved_time = zeros(0) 
    saved_iters = zeros(0)

    VBparams.wallClockTime  = 0.0;
    #VB loop
    numDocsSoFar = 0;
    totalTime = 0;
    while VBparams.wallClockTime < time_limit
        tic_time = tic();
        numDocsSoFar += 1
        VBparams.iter = numDocsSoFar;
        if mod(numDocsSoFar, 1000) == 0
            println("Document iteration $(numDocsSoFar).");
        end
        docInd = mod(numDocsSoFar -1, length(documents)) + 1;
        docLength = documents[docInd].docLength;
        if docLength == 0
            continue;
        end
        
        VBparams = learnThetaForDoc_Sparse(VBparams, documents, docInd, burnInPerDoc, tau2, kappa2, meanchangeThreshold);
        VBparams, wordTopicCounts_hat, topicCounts_hat = updateBoth_SparseFull(VBparams, documents, docInd, tau2 + burnInPerDoc .* docLength, kappa2, wordTopicCounts_hat, topicCounts_hat);
        if mod(numDocsSoFar, minibatchSize) == 0 

            #compute effective stepsize to account for minibatches
            stepSize = 1 - (1-stepSize)^minibatchSize 

            #Do a minibatch update
            topicCounts = VBparams.topicCounts
            wordTopicCounts = VBparams.wordTopicCounts
            documentTopicCounts = VBparams.documentTopicCounts
            for topic = 1:numTopics
                for word = 1:numWords
                    wordTopicCounts[word,topic] = (1 - stepSize) .* wordTopicCounts[word,topic] + 
                            stepSize .* miniBatchesPerCorpus .* wordTopicCounts_hat[word,topic];
                end
                topicCounts[topic] = (1 - stepSize) .* topicCounts[topic] + stepSize .* 
                                     miniBatchesPerCorpus .* topicCounts_hat[topic];

            end
            
            wordTopicCounts_hat *= 0;
            topicCounts_hat *= 0;
            stepSize = scale ./(numDocsSoFar + tau)^kappa;
        end
        
        VBparams.wallClockTime = VBparams.wallClockTime + toq();
        ##########debug##########
        if debugTopics
            if mod(numDocsSoFar, 1000) == 0
                for t = 1:min(numTopics,100)
                    topWords, topWordProbs = getImportantWordsInTopic(VBparams.wordTopicCounts[:,t], dict, 10);
                    println(topWords)
                end
            end
        end


        if saveParams && VBparams.wallClockTime > savepoints[saveInd]
            push!(saved_topics, copy(VBparams.wordTopicCounts))
            push!(saved_alphas, VBparams.alpha)
            push!(saved_time, VBparams.wallClockTime)
            push!(saved_iters, numDocsSoFar)
            saveInd += 1;
        end
    end
    VBparams, saved_iters, saved_time, saved_topics, saved_alphas
end



function initialize(documents, alpha, eta, numWords, numTokens)
    numDocuments = length(documents);
    numTopics = length(alpha);

    VBparams = collapsed_VB_params_type(alpha, eta, numTopics, numWords)
    
    #pick a random document and initialize total counts based on its length
    VBparams.topicCounts *= numTokens ./ numTopics;
    VBparams.wordTopicCounts *= numTokens ./ (numWords .* numTopics);
    
    #initialize document distributions
    VBparams.documentTopicCounts = zeros(length(documents), numTopics);
    for i = 1:length(documents)
        docLength = documents[i].docLength;
        documentTopicCounts = rand(1, numTopics);
        VBparams.documentTopicCounts[i,:] = (documentTopicCounts ./ sum(documentTopicCounts)) * docLength;
    end
    VBparams 
end

function stochasticCVB0_testMode(documents::Array{SparseDocument,1},
                                                            VBparams,
                                                            burnInPerDoc::Int,
                                                            tau2::Float64,
                                                            kappa2::Float64)
#Stochastic Collapsed Variational Bayes for LDA
#Test mode - learns theta while keeping the topics fixed.
#@author Jimmy Foulds
    VBparams = deepcopy(VBparams)

    meanchangeThreshold = 0;

    #VB loop
    for numDocsSoFar = 1:length(documents)
        if mod(numDocsSoFar, 1000) == 0
            println("Document iteration $(numDocsSoFar).");
        end
        docInd = mod(numDocsSoFar -1, length(documents)) + 1;
        docLength = documents[docInd].docLength;
        if docLength == 0
            continue;
        end
        
        VBparams = learnThetaForDoc_Sparse(VBparams, documents, docInd, burnInPerDoc, tau2, kappa2, meanchangeThreshold);

    end
    VBparams.documentTopicCounts
end
