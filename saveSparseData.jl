include("definitions.jl")

function saveSparseData(sparseDataFilename::AbstractString, documents::Array{SparseDocument,1})
#save a dataset in the sparse document format
#one document per line, consisting of pairs of token index and token count, space separated.
    f = open(sparseDataFilename, "w");
    for doc in documents
        for i = 1:length(doc.tokenIndex)
           write(f, "$(doc.tokenIndex[i]) $(doc.tokenCount[i])");
           if i < length(doc.tokenIndex)
               write(f, " ");
           end
        end
        write(f,"\n");
    end
    close(f);
end
