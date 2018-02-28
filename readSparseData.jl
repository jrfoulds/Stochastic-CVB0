include("definitions.jl")

function readSparseData(sparseDataFilename::AbstractString, dictionaryFilename::AbstractString)
#read a dataset in sparse format. One line per document, consisting of pairs of token index and token count, space separated.
    f = open(dictionaryFilename);
        dictionary = readlines(f); #an array of strings
    close(f);
    
    f = open(sparseDataFilename);
    documents = Array{SparseDocument}(0);
    for ln in eachline(f)
        splitLn = split(ln);
        wordCounts = Array{Int64}(length(splitLn));
        docLength = 0;
        numTokensFound = convert(Int64,floor(length(splitLn) / 2));
        tokenIndex = zeros(Int64, numTokensFound);
        tokenCount = zeros(Int64, numTokensFound);
        ind = 1;
        for i = 1:2:length(splitLn)
           tokenIndex[ind] = parse(Int64, splitLn[i]);
           tokenCount[ind] = parse(Int64, splitLn[i + 1]);
           docLength = docLength + tokenCount[ind];
           ind = ind + 1;
        end
        push!(documents, SparseDocument([], tokenIndex, tokenCount, docLength));
    end
    close(f);
    
    documents, dictionary
end
