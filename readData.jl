include("definitions.jl")

function readData(dataFilename::AbstractString, dictionaryFilename::AbstractString)
#read a dataset in non-sparse format. One line per document, word indices (with one-based indexing), space separated.
    f = open(dictionaryFilename);
        dictionary = readlines(f); #an array of strings
    close(f);
    
    f = open(dataFilename);
    documents = Array{Document}(0);
    for ln in eachline(f)
        splitLn = split(ln);
        wordVector = Array{Int64}(length(splitLn));
        for i = 1:length(splitLn)
           wordVector[i] = parse(Int64, splitLn[i]);
        end
        push!(documents, Document(wordVector));
    end
    close(f);
    
    documents, dictionary
end
