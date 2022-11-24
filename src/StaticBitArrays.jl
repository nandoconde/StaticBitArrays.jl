module StaticBitArrays


# Base imports to add methods
import Base: @propagate_inbounds
import Base: getindex, setindex!, size, similar, vec, show, length, convert, # promote_op, promote_rule, 
             map, map!, reduce, mapreduce, broadcast,
             broadcast!, conj, hcat, vcat, ones, zeros, one, reshape, fill, fill!, inv,
             iszero, sum, prod, count, any, all, minimum, maximum, extrema,
             copy, read, read!, write, reverse
             zero, count_ones, iszero, empty!, isempty, eltype, iterate, isless, isequal, hash,
             ~, +, |, xor, in, ==

# import Base.SimdLoop: simd_index, simd_inner_length, simd_outer_range
import Base.Broadcast: broadcasted, materialize!, broadcastable



## Ancillary functions 
@inline _div64(l) = l >> 6
@inline _mul64(l) = l << 6
@inline num_bit_chunks(n::Int) = _div64(n + 63)
@inline _mod64(l) = l & 63
const _mask64 = ~UInt64(0)
const _zero64 = UInt64(0)
const _right64 = UInt64(1)
const _left64 = _mask64 ⊻ (_mask64 >>> 1)
@inline _msk_ini(l::Int) = _mask64 << _mod64(-l)
@inline _msk_ini(B::StaticBitArray) = _msk_ini(length(B))
@inline _msk_end(l::Int) = _mask64 >>> _mod64(-l)
@inline _msk_end(B::StaticBitArray) = _msk_end(length(B))
# Very unsafe, assumes that there are 64 elements indexable with 1-based from 1 to 64
@inline function _gather64(v::AbstractVector{Bool})
    x = _zero64
    @inbounds for i in 1:64
        x = x >>> 1
        v[i] && (x |= _left64)
    end
end

# Very unsafe, assumes that there are 8 elements indexable with 1-based from 1 to 8
@inline function _gather8(v::AbstractVector{Bool})
    x = _zero64
    @inbounds for i in 1:8
        x = x >>> 1
        v[i] && (x |= _left64)
    end
end

# Very unsafe, assumes that there are N < 64 elements indexable with 1-based from 1 to N
@inline function _gatherN(v::AbstractVector{Bool}, N)
    x = _zero64
    @inbounds for i in 1:N
        x = x >>> 1
        v[i] && (x |= _left64)
    end
end

## Type and constructors
# TODO: uncomment once all vector methods have been implemented
# abstract type StaticBitArray{N} <: AbstractVector{Bool} where N end
abstract type StaticBitArray{N} end
"""

The empty part of the last chunk is guaranteed to be all zeros.

"""
struct SBitVector{N} <: StaticBitArray{N}
    chunks::NTuple{N,UInt64}
    len::Int
    
    # Inner constructor ensures that chunks are just enough to hold given length
    function SBitVector{N}(chunks::NTuple{N,UInt64}, L::Int) where N
        if L < 0
            throw(ArgumentError("length must be ≥ 0, got $L"))
        elseif L == 0
            (chunks == NTuple{0,UInt64}(0))  && return new{0}(chunks, L)
            throw(ArgumentError("length 0 needs an empty tuple"))
        else
            n = num_bit_chunks(L)
            n == N && return new{N}(chunks, L)
            throw(ArgumentError("length $L needs $n chunks, not $N"))
        end
    end
end

mutable struct MBitVector{N} <: StaticBitArray{N}
    chunks::NTuple{N,UInt64}
    len::Int
    
    # Inner constructor ensures that chunks are just enough to hold given length
    function MBitVector{N}(chunks::NTuple{N,UInt64}, L::Int) where N
        if L < 0
            throw(ArgumentError("length must be ≥ 0, got $L"))
        elseif L == 0
            isa(chunks, NTuple{0,UInt64}) && return new{0}(chunks, L)
            throw(ArgumentError("length 0 needs an empty tuple"))
        else
            n = num_bit_chunks(L)
            n == N && return new{N}(chunks, L)
            throw(ArgumentError("length $L needs $n chunks, not $N"))
        end
    end
end

# Types as symbols for quickly metaprogramming most methods
sbv_types = (:SBitVector, :MBitVector)
# Constructor with empty parameter
for t in sbv_types
    @eval $(t)(chunks::NTuple{N,UInt64}, L::Int) where N = $(t){N}(chunks, L)
end
# SBitVector(chunks::NTuple{N,UInt64}, L::Int) where N = SBitVector{N}(chunks, L)
# MBitVector(chunks::NTuple{N,UInt64}, L::Int) where N = MBitVector{N}(chunks, L)

# Constructor from vector (reinterpreted)
for t in sbv_types
    @eval begin
        function $(t)(chunks::Vector{Union{UInt64,Int64}}, L::Int)
            N = num_bit_chunks(L)
            return $(t)(reinterpret(NTuple{N,UInt64}, chunks), L)
        end
    end
end
# function SBitVector(chunks::Vector{Union{UInt64,Int64}}, L::Int)
#     N = num_bit_chunks(L)
#     return SBitVector(reinterpret(NTuple{N,UInt64}, chunks), L)
# end
# function MBitVector(chunks::Vector{Union{UInt64,Int64}}, L::Int)
#     N = num_bit_chunks(L)
#     return MBitVector(reinterpret(NTuple{N,UInt64}, chunks), L)
# end

# Empty constructors
for t in sbv_types
    @eval $(t)() = $(t)(NTuple{0,UInt64}(), 0)
end
# SBitVector() = SBitVector(NTuple{0,UInt64}, 0)
# MBitVector() = MBitVector(NTuple{0,UInt64}, 0)

# Constructor with Tuple and vector of bools
# TODO: help needed to avoid allocations
#    - Maybe iterate the input tuple in units of 64 and then collect. Does that allocate?
#    - Create the final structure and unsafe_store interate them
for t in sbv_types
    @eval begin
        function $(t)(bt::Tuple{Vararg{Bool}})
            L = length(bt)
            # Maybe unnecesary
            if L == 0
                return $(t)()
            end
            N = num_bit_chunks(L)
            uiv = zeros(UInt, N)
            @inbounds for i in 1:L
                j = num_bit_chunks(i)
                if bt[i]
                    uiv[j] |= (UInt(1) << (_mod64(i - 1)))
                end
            end
            sbv = $(t)(NTuple{N, UInt}(uiv), L)
        end
    end
end

# function SBitVector(bt::Tuple{Vararg{Bool}})
#     L = length(bt)
#     # Maybe unnecesary
#     if L == 0
#         return SBitVector()
#     end
#     N = num_bit_chunks(L)
#     uiv = zeros(UInt, N)
#     @inbounds for i in 1:L
#         j = num_bit_chunks(i)
#         if bt[i]
#             uiv[j] |= (UInt(1) << (_mod64(i - 1)))
#         end
#     end
#     sbv = SBitVector(NTuple{N, UInt}(uiv), L)
# end
for t in sbv_types
    @eval $(t)(bv::Vector{Bool}) = $(t)(Tuple(bv))
end
# SBitVector(bv::Vector{Bool}) = SBitVector(Tuple(bv))




## Utility functions
@inline length(a::StaticBitArray{N}) = a.len
@inline size(a::StaticBitArray) = (a.len,)
@inline function size(a::StaticBitArray, d::Integer)
    d < 1 && throw_boundserror(size(a), d)
    ifelse(d == 1, a.len, 1)
end
isassigned(a::StaticBitArray, i::Int) = 1 <= i <= length(a)
IndexStyle(::Type{<:StaticBitArray}) = IndexLinear()
@inline get_chunks_id(i::Int) = (num_bit_chunks(i), _mod64(i-1))



## Zero
# Constructor for all zeros
zero(::Type{SBitVector{N}}) where N = SBitVector(ntuple(i->UInt64(0), Val{N}()), _mul64(N))

# Check for zero
@inline function iszero(a::StaticBitArray{N}) where N
    s = UInt64(0)
    @inbounds for i = 1:N
        s |= a.chunks[i]
    end
    return iszero(s)
end

# Emptiness if treated as vector
@inline isempty(a::StaticBitArray) = iszero(a)

# TODO: benchmark to see which is faster
# Zeros are counted by counting ones, to facilitate the operation in the last chunk
@inline count_zeros(a::StaticBitArray{N}) where N = a.len - count_ones(a)
# Alt:
# @inline function count_zeros(a::StaticBitArray{N}) where N 
#     s = 0
#     @inbounds for i = 1:(N-1)
#         s+= count_zeros(a.chunks[i])
#     end
#     @inbounds s += count_zeros(a.chunks[N] | _left_ones(_mod64(length(a))))
#     return s
# end


## One
@inline function count_ones(a::StaticBitArray{N}) where N 
    s = 0
    @inbounds for i = 1:N 
        s+= count_ones(a.chunks[i])
    end
    return s
end


## Iteration Protocol
# _blsr(x) = x & (x-Int64(1))
@inline eltype(::StaticBitArray) = Bool

# First index is 1 to be consistent with usual array indexing, instead of Base.BitVector
@inline function iterate(a::SBitVector{N}, s::Int = 1) where N
    # Pre checks
    N > 0 || s > 1 || s <= length(a) || return nothing

    # Return current bool and next state
    # TODO: Benchmark with @inbounds
    return ((a.chunks[num_bit_chunks(s)] & (UInt64(1) << _mod64(s-1))) != 0, s + 1)
end



## Bitwise operations
@inline Base.:&(L::StaticBitArray{N}, R::StaticBitArray{N}) where N =  SBitSet(ntuple(i->(L.chunks[i] & R.chunks[i]), Val{N}()))
@inline Base.:|(L::UBitSet{N}, R::UBitSet{N}) where N =  SBitSet(ntuple(i->(L.chunks[i] | R.chunks[i]), Val{N}()))
@inline xor(L::UBitSet{N}, R::UBitSet{N}) where N =  SBitSet(ntuple(i->xor(L.chunks[i], R.chunks[i]), Val{N}()))
@inline ~(a::UBitSet{N}) where N = SBitSet(ntuple(i->~a.chunks[i], Val{N}()))



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
