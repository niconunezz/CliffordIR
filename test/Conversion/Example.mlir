// RUN: cliff-opt %s -split-input-file --convert-cliffGPU-to-llvm | FileCheck %s

#layout = #clg.linear<{register = [], lane = [ [1], [2], [4], [8], [16]], warp = [[32]], block = []}>

#space = #cliff.algebra<{p=2, q=0, r=1}>

cliff.func @geo_prod_scalar_case(%arg0 : !cliff.ptr<f32>, %arg1 : !cliff.ptr<f32>, %pointer : !cliff.ptr<f32>) {
    
    %t0 = clg.load %arg0 : !cliff.ptr<f32> -> tensor<64x!cliff.multivector<MASK_A, unknown, f32, #space>, #layout>
    %t1 = clg.load %arg1 : !cliff.ptr<f32> -> tensor<64x!cliff.multivector<MASK_B, unknown, f32, #space>, #layout>

    %0 = cliff.geo_prod %t0, %t1 :
        tensor<64x!cliff.multivector<MASK_A, unknown, f32, #space>, #layout> *
        tensor<64x!cliff.multivector<MASK_B, unknown, f32, #space>, #layout> ->
        tensor<64x!cliff.multivector<MASK_C, unknown, f32, #space>, #layout>
    
    clg.store %pointer, %0 : !cliff.ptr<f32>, tensor<64x!cliff.multivector<MASK_C, unknown, f32, #space>, #layout>
    cliff.ret
}