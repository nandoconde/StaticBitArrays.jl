using TestItems
using TestItemRunner
import StaticBitArrays

@testitem "Ancillary functions" begin
    ## Ancillary functions 
    # _div64
    @test StaticBitArrays._div64(0) == 0
    @test StaticBitArrays._div64(63) == 0
    @test StaticBitArrays._div64(64) == 1
    #_mul64
    @test StaticBitArrays._div64(0) == 0
    @test StaticBitArrays._div64(1) == 64
    @test StaticBitArrays._div64(2) == 128
    # num_bit_chunks
    @test StaticBitArrays.num_bit_chunks(0) == 0
    @test StaticBitArrays.num_bit_chunks(1) == 1
    @test StaticBitArrays.num_bit_chunks(64) == 1
    @test StaticBitArrays.num_bit_chunks(65) == 2
    # _mod64
    @test StaticBitArrays._mod64(0) = 0
    @test StaticBitArrays._mod64(-1) = 63
    @test StaticBitArrays._mod64(-2) = 62
    @test StaticBitArrays._mod64(1) = 1
    @test StaticBitArrays._mod64(63) = 63
    @test StaticBitArrays._mod64(64) = 0
    # _msk_ini
    @test StaticBitArrays._msk_ini(60) == 0xFFFFFFFFFFFFFFF0
    @test StaticBitArrays._msk_ini(4) == 0xF000000000000000
    # _msk_end
    @test StaticBitArrays._msk_end(4) == 0x000000000000000F
    @test StaticBitArrays._msk_end(60) == 0x0FFFFFFFFFFFFFFF
end

@testitem "Struct creation" begin


end