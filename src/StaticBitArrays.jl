module StaticBitArrays


# Heavily used functions
@inline _div64(l) = l >> 6
@inline _div64_1(l) = (l >> 6)
@inline _mod64(l) = l & 63

# Untyped length
struct SBitVector{N} <: AbstractVector{Bool}
    chunks::NTuple{N,UInt64}
    length::Int
    
    # Inner constructor ensures that chunks are just enough to hold given length
    function SBitVector{N}(chunks::NTuple{N,UInt64}, length::Int)
        if length < 0
            throw(ArgumentError("length must be \ge 0"))


    end
end



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

greet() = print("Hello World!")

end # module StaticBitArrays
