include("definitions.jl")

#Learn variational parameters for theta from one document, leaving topics fixed.
#Non-sparse version, for reference, not used.
function learnThetaForDoc(VBparams::collapsed_VB_params_type, documents::Array{Document,1}, docInd::Int, burnInPerDoc::Int, tau2::Float64, kappa2::Float64, meanchangeThreshold::Float64)
    oldDTC = VBparams.documentTopicCounts[docInd,:];
    docLength = length(documents[docInd].wordVector);
    probs::Array{Float64,1} = Array(Float64, length(VBparams.alpha))


    eta = VBparams.eta
    alpha = VBparams.alpha
    topicCounts = VBparams.topicCounts
    etaSum = VBparams.etaSum
    wordTopicCounts = VBparams.wordTopicCounts
    documentTopicCounts = VBparams.documentTopicCounts
    K = length(alpha)


    for burn=1:burnInPerDoc
        for wordInd = 1:docLength
            word::Int = documents[docInd].wordVector[wordInd];
            stepSize2::Float64 = (wordInd + (burn-1) .* docLength + tau2)^-kappa2;

            for topic = 1:K
                probs[topic] = (wordTopicCounts[word, topic] + eta[word]) .* 
                        (documentTopicCounts[docInd,topic] + alpha[topic]) ./ 
                        (topicCounts[topic] + etaSum);
            end

            probs /= sum(probs);
            #update params for theta
            for topic = 1:K
                documentTopicCounts[docInd,topic] = (1 - stepSize2) * documentTopicCounts[docInd,topic] + stepSize2 .* docLength .* probs[topic];
            end
        end
    end
    VBparams 
end

#Learn variational parameters for theta and phi from one document
#Non-sparse version, for reference, not used.
function learnTopicsForDoc(VBparams::collapsed_VB_params_type, documents::Array{Document,1}, docInd, tau2, kappa2, wordTopicCounts_hat, topicCounts_hat)
    docLength = length(documents[docInd].wordVector);
    probs::Array{Float64,1} = Array(Float64, length(VBparams.alpha))

    eta = VBparams.eta
    alpha = VBparams.alpha
    topicCounts = VBparams.topicCounts
    etaSum = VBparams.etaSum
    wordTopicCounts = VBparams.wordTopicCounts
    documentTopicCounts = VBparams.documentTopicCounts
    K = length(alpha)

    for wordInd = 1:docLength
        word::Int = documents[docInd].wordVector[wordInd];
        stepSize2::Float64 = (wordInd + tau2)^-kappa2;

        for topic = 1:K
            probs[topic] = (wordTopicCounts[word, topic] + eta[word]) .* 
                    (documentTopicCounts[docInd,topic] + alpha[topic]) ./ 
                    (topicCounts[topic] + etaSum);
        end
        
        probs /= sum(probs);

        #update sufficient statistics
        for topic = 1:K
            VBparams.documentTopicCounts[docInd,topic] = (1 - stepSize2) * documentTopicCounts[docInd,topic] + stepSize2 .* docLength .* probs[topic];
            wordTopicCounts_hat[word, topic] = wordTopicCounts_hat[word, topic] + probs[topic];
            topicCounts_hat[topic] += probs[topic];
        end
    end
    VBparams, wordTopicCounts_hat, topicCounts_hat 
end

#Learn variational parameters for theta, leaving topics fixed.
#sparse, with clumping
function learnThetaForDoc_Sparse(VBparams, documents, docInd, burnInPerDoc, tau2, kappa2, meanchangeThreshold)
    oldDTC = VBparams.documentTopicCounts[docInd,:];
    docLength = documents[docInd].docLength;
    numDistinctTokens = length(documents[docInd].tokenIndex);
    stepSizeCounter = 1;
    stepSize2 = (stepSizeCounter + tau2)^-kappa2;

    probs::Array{Float64,1} = Array{Float64}(length(VBparams.alpha))

    eta = VBparams.eta
    alpha = VBparams.alpha
    topicCounts = VBparams.topicCounts
    etaSum = VBparams.etaSum
    wordTopicCounts = VBparams.wordTopicCounts
    documentTopicCounts = VBparams.documentTopicCounts
    K = length(alpha)

    for burn=1:burnInPerDoc
        for tokenInd = 1:numDistinctTokens
            word = documents[docInd].tokenIndex[tokenInd];
            
            for topic = 1:K
                probs[topic] = (wordTopicCounts[word, topic] + eta[word]) .* 
                        (documentTopicCounts[docInd,topic] + alpha[topic]) ./ 
                        (topicCounts[topic] + etaSum);
            end

            sum_probs = sum(probs)
            
            #update params for theta
            tokenCountInDoc = documents[docInd].tokenCount[tokenInd];

            left_step = (1 - stepSize2)^tokenCountInDoc
            right_step = 1-(1-stepSize2)^tokenCountInDoc
            for topic = 1:K
                probs[topic] /= sum_probs;
                documentTopicCounts[docInd,topic] = left_step .* documentTopicCounts[docInd,topic] + 
                                                    docLength .* probs[topic] .* right_step;
            end

            stepSizeCounter = stepSizeCounter + tokenCountInDoc;
            stepSize2 = (stepSizeCounter + (burn-1) .* docLength + tau2)^-kappa2;
        end
        meanchange = mean(abs.(VBparams.documentTopicCounts[docInd,:] - oldDTC));
        if meanchange < meanchangeThreshold
            #fprintf("converged after #d iterations\n", burn);
            break;
        end
        oldDTC = VBparams.documentTopicCounts[docInd,:];
    end
    VBparams 
end

#Learn variational parameters for theta and phi
#sparse, with clumping
function updateBoth_SparseFull(VBparams, documents, docInd, tau2, kappa2, wordTopicCounts_hat, topicCounts_hat)
    docLength = documents[docInd].docLength;
    numDistinctTokens = length(documents[docInd].tokenIndex);
    stepSizeCounter = 1;
    stepSize2 = (stepSizeCounter + tau2)^-kappa2;

    probs::Array{Float64,1} = Array{Float64}(length(VBparams.alpha))

    eta = VBparams.eta
    alpha = VBparams.alpha
    topicCounts = VBparams.topicCounts
    etaSum = VBparams.etaSum
    wordTopicCounts = VBparams.wordTopicCounts
    documentTopicCounts = VBparams.documentTopicCounts
    K = length(alpha)

    for tokenInd = 1:numDistinctTokens
        word = documents[docInd].tokenIndex[tokenInd];

        #CVB0 estimate of gamma for current token
        for topic = 1:K
            probs[topic] = (wordTopicCounts[word, topic] + eta[word]) .* 
                    (documentTopicCounts[docInd,topic] + alpha[topic]) ./ 
                    (topicCounts[topic] + etaSum);
        end

        sum_probs = sum(probs);

        tokenCountInDoc = documents[docInd].tokenCount[tokenInd];

        left_step = (1 - stepSize2)^tokenCountInDoc
        right_step = 1-(1-stepSize2)^tokenCountInDoc

        for topic = 1:K
            probs[topic] /= sum_probs
            documentTopicCounts[docInd,topic] = left_step .* documentTopicCounts[docInd,topic] + docLength .* probs[topic] .* right_step;
            wordTopicCounts_hat[word, topic] = wordTopicCounts_hat[word, topic] + probs[topic] * tokenCountInDoc;
            topicCounts_hat[topic] += probs[topic]  * tokenCountInDoc;
        end

 
        stepSizeCounter = stepSizeCounter + tokenCountInDoc;
        stepSize2 = (stepSizeCounter + tau2)^-kappa2;
    end
    VBparams, wordTopicCounts_hat, topicCounts_hat
end
