// RUN: cliff-opt %s -split-input-file --convert-cliffGPU-to-llvm | FileCheck %s

#layout = #clg.linear<LAYOUT>

#space = #cliff.algebra<{p=2, q=0, r=1}>

cliff.func @geo_prod(%arg0 : !clg.ptr<f32>, %arg1 : !clg.ptr<f32>, %pointer : !clg.ptr<f32>) {
    
    %t0 = clg.load %arg0 : !clg.ptr<f32> -> tensor<NUM_ELSx!cliff.multivector<MASK_A, unknown, f32, #space>, #layout>
    %t1 = clg.load %arg1 : !clg.ptr<f32> -> tensor<NUM_ELSx!cliff.multivector<MASK_B, unknown, f32, #space>, #layout>

    %0 = cliff.geo_prod %t0, %t1 :
        tensor<NUM_ELSx!cliff.multivector<MASK_A, unknown, f32, #space>, #layout> *
        tensor<NUM_ELSx!cliff.multivector<MASK_B, unknown, f32, #space>, #layout> ->
        tensor<NUM_ELSx!cliff.multivector<MASK_C, unknown, f32, #space>, #layout>
    
    %1 = cliff.reverse %0 :  tensor<NUM_ELSx!cliff.multivector<MASK_C, unknown, f32, #space>, #layout> ->  tensor<NUM_ELSx!cliff.multivector<MASK_C, unknown, f32, #space>, #layout>
    
    clg.store %pointer, %1 : !clg.ptr<f32>, tensor<NUM_ELSx!cliff.multivector<MASK_C, unknown, f32, #space>, #layout>
    cliff.ret
}
