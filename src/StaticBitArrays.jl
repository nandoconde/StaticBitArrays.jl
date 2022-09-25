module StaticBitArrays


# Heavily used functions
@inline _div64(l) = l >> 6
@inline _mul64(l) = l << 6
@inline num_bit_chunks(n::Int) = _div64(n+63)
@inline _mod64(l) = l & 63
@inline _left_ones(l) = 0xFFFFFFFFFFFFFFFF << (64-l)


## Type and constructors
"""

The empty part of the last chunk is guaranteed to be all zeros.

"""
struct SBitVector{N} <: AbstractVector{Bool}
    chunks::NTuple{N,UInt64}
    len::Int
    
    # Inner constructor ensures that chunks are just enough to hold given length
    function SBitVector{N}(chunks::NTuple{N,UInt64}, L::Int) where N
        if L < 0
            throw(ArgumentError("length must be ≥ 0, got $L"))
        elseif L == 0
            isa(chunks, NTuple{0,UInt64}) && return new(chunks, L)
            throw(ArgumentError("length 0 needs an empty tuple"))
        else
            n = num_bit_chunks(L)
            n == N && return new(chunks, L)
            throw(ArgumentError("length $L needs $n chunks, not $N"))
        end
    end
end

# Constructor with empty parameter
SBitVector(chunks::NTuple{N,UInt64}, L::Int) where N = SBitVector{N}(chunks, L)

# Constructor from vector (reinterpreted)
function SBitVector(chunks::Vector{Union{UInt,Int}}, L::Int)
    N = num_bit_chunks(L)
    return SBitVector(reinterpret(NTuple{N,UInt}, chunks), L)
end


## Length (to avoid field accessing)
@inline length(a::SBitVector) = a.len


## Zero
# Constructor for all zeros
zero(::Type{SBitVector{N}}) where N = SBitVector(ntuple(i->UInt64(0), Val{N}()), _mul64(N))

# Check for zero
@inline function iszero(a::SBitVector{N}) where N
    s = UInt64(0)
    @inbounds for i = 1:N
        s |= a.chunks[i]
    end
    return iszero(s)
end

# Emptiness if treated as vector
@inline isempty(a::SBitVector) = iszero(a)

# TODO: benchmark to see which is faster
# Zeros are counted by counting ones, to facilitate the operation in the last chunk
@inline count_zeros(a::SBitVector{N}) where N = a.len - count_ones(a)
# Alt:
# @inline function count_zeros(a::SBitVector{N}) where N 
#     s = 0
#     @inbounds for i = 1:(N-1)
#         s+= count_zeros(a.chunks[i])
#     end
#     @inbounds s += count_zeros(a[N] | _left_ones(_mod64(length(a))))
#     return s
# end


## One
@inline function count_ones(a::SBitVector{N}) where N 
    s = 0
    @inbounds for i = 1:N 
        s+= count_ones(a.chunks[i])
    end
    return s
end


## Iteration Protocol
# _blsr(x) = x & (x-Int64(1))
eltype(::SBitVector) = Bool

@inline function iterate(a::SBitVector{N}) where N
    N > 0 || return nothing
    return iterate(a, (1, @inbounds a.chunks[1]))
end

# TODO: continue here
@inline function iterate(a::SBitVector{N}, s) where N
    chunks = a.chunks
    i1, c = s
    while c==0
        i1 % UInt >= N % UInt && return nothing
        i1 += 1
        @inbounds c = chunks[i1]
    end
    tz = trailing_zeros(c) + 1
    c = _blsr(c)
    return ((i1-1)<<6 + tz, (i1, c))
end


# BitVector
# Conversion to BitVector
function convert(::Type{BitVector}, a::SBitVector{N})  where N
    res = BitVector(undef, length(a))
    @inbounds for i = 1:N
        res.chunks[i] = a.chunks[i]
    end
    res
end

# Conversion from BitVector
convert(::Type{SBitVector}, a::BitVector) = SBitVector(ntuple(i->a.chunks[i], a.chunks), length(a))






end # module StaticBitArrays
