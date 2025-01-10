# These functions create Generalized Orthonormal Basis Functions (GOBFs)
# that define the filter bank in the neuron model. 
# The filter bank maps voltage v(t) into a high dimensional vector
# w(t). Because the basis filter transfer functions are orthonormal,
# the activity in this higher dimensional space is very "rich". 
# See Burghi, Schoukens and Sepulchre (2020) for more info.

function cascade(A1,B1,C1,D1,A2,B2,C2,D2)
    # Given two realizations sys1=(A1,B1,C1,D1) and sys2=(A2,B2,C2,D2), returns
    # the (A,B,C,D) realization of the cascade system ->[sys1]->[sys2]->
        if isempty(A1)
            return (A2,B2,C2,D2)
        else
            A = [A1  zeros(eltype(A1),size(A1,1),size(A2,2)); 
                B2*C1 A2];
            B = [B1 ; B2*D1];
            C = [D2*C1 C2];
            D = D2*D1;
            return (A,B,C,D)
        end
end

function GOBFRealization(p::V) where V <: AbstractVector
    A, B, C, D = eltype(V)[], eltype(V)[], eltype(V)[], eltype(V)[]    
    for i in eachindex(p)
        Ai = p[i]
        Bi = sqrt(1-abs(p[i])^2)
        Ci = Bi
        Di = -conj(p[i])
        (A,B,C,D) = cascade(A,B,C,D,Ai,Bi,Ci,Di)
    end
    return A,reshape(B,size(B,1),size(B,2)) #force B to be of type matrix
end

function GOBFRealization(p::Vector{V}) where V <: AbstractVector
    Avec,Bvec = [],[]
    totalstates = 0
    for n = eachindex(p)
        A, B, C, D = [], [], [], []    
        for i in eachindex(p[n])
            Ai = p[n][i]
            Bi = sqrt(1-abs(p[n][i])^2)
            Ci = Bi
            Di = -conj(p[n][i])
            (A,B,C,D) = cascade(A,B,C,D,Ai,Bi,Ci,Di)
        end
        push!(Avec,A)
        push!(Bvec,B)
        totalstates += length(p[n])
    end
    A = zeros(eltype(V),totalstates,totalstates) 
    B = zeros(eltype(V),totalstates,1)
    diag_index = 1
    for n = 1:length(p)
        A[diag_index:diag_index+length(p[n])-1,diag_index:diag_index+length(p[n])-1] = Avec[n]   
        B[diag_index:diag_index+length(p[n])-1,1] = Bvec[n]
        diag_index += length(p[n])
    end
    return A,B
end

function DiagRealization(p::V) where V <: AbstractVector
    A = Diagonal(p)
    B = 1 .- p
    return A,reshape(B,size(B,1),size(B,2)) #force B to be of type matrix
end

# function CTGOBFRealization(p)
#     A, B, C, D = Float64[], Float64[], Float64[], Float64[]    
#     for i in eachindex(p)
#         Ai = p[i]
#         Bi = sqrt(2*real(-p[i]))
#         Ci = -Bi
#         Di = 1
#         (A,B,C,D) = cascade(A,B,C,D,Ai,Bi,Ci,Di)
#     end
#     return A,reshape(B,size(B,1),size(B,2)) #force B to be of type matrix
# end