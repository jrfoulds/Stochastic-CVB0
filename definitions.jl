import Base.length
import Base.ndims

type Document
    wordVector::Array{Int64,1}
end

type SparseDocument
    wordVector::Array{Int64,1}
    tokenIndex::Array{Int64,1}
    tokenCount::Array{Int64,1}
    docLength::Int64
end

type collapsed_VB_params_type
    alpha::Array{Float64,2}
    eta::Array{Float64,2}
    etaSum::Float64
    numTopics::Int64
    topicCounts::Array{Float64,2}
    wordTopicCounts::Array{Float64,2}
    iter::Int
    wallClockTime::Float64
    documentTopicCounts::Array{Float64,2}
    tokenTopicDistributions::Array{Array{Float64,2},1}
end 

collapsed_VB_params_type(alpha, eta, numTopics, numWords) = 
    collapsed_VB_params_type(alpha, eta, sum(eta), numTopics, ones(1,numTopics),
                             ones(numWords,numTopics), 0, 0.0, ones(0,0),Array{Array{Float64,2}}(0)) 
length(x::collapsed_VB_params_type) = 1
ref(x::collapsed_VB_params_type,i::Int) = x
ndims(x::collapsed_VB_params_type) = 0

