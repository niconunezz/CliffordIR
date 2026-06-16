// RUN: cliff-opt %s -split-input-file --convert-cliffGPU-to-llvm | FileCheck %s

#layout = #clg.linear<{register = [], lane = [ [1], [2], [4], [8], [16]], warp = [[32]], block = []}>

#space = #cliff.algebra<{p=2, q=0, r=1}>

cliff.func @geo_prod(%arg0 : !cliff.ptr<f32>, %arg1 : !cliff.ptr<f32>, %pointer : !cliff.ptr<f32>) {
    
    %t0 = clg.load %arg0 : !cliff.ptr<f32> -> tensor<64x!cliff.multivector<MASK_A, unknown, f32, #space>, #layout>
    %t1 = clg.load %arg1 : !cliff.ptr<f32> -> tensor<64x!cliff.multivector<MASK_B, unknown, f32, #space>, #layout>

    %0 = cliff.geo_prod %t0, %t1 :
        tensor<64x!cliff.multivector<MASK_A, unknown, f32, #space>, #layout> *
        tensor<64x!cliff.multivector<MASK_B, unknown, f32, #space>, #layout> ->
        tensor<64x!cliff.multivector<MASK_C, unknown, f32, #space>, #layout>
    
    %1 = cliff.reverse %0 :  tensor<64x!cliff.multivector<MASK_C, unknown, f32, #space>, #layout> ->  tensor<64x!cliff.multivector<MASK_C, unknown, f32, #space>, #layout>
    
    clg.store %pointer, %1 : !cliff.ptr<f32>, tensor<64x!cliff.multivector<MASK_C, unknown, f32, #space>, #layout>
    cliff.ret
}

cliff.func @rotation(%arg0 : !cliff.ptr<f32>, %arg1 : !cliff.ptr<f32>, %pointer : !cliff.ptr<f32>) {
    
    %t0 = clg.load %arg0 : !cliff.ptr<f32> -> tensor<64x!cliff.multivector<105, unknown, f32, #space>, #layout>
    %t1 = clg.load %arg1 : !cliff.ptr<f32> -> tensor<64x!cliff.multivector<1, unknown, f32, #space>, #layout>

    %0 = cliff.rotate %t0, %t1 :
        tensor<64x!cliff.multivector<105, unknown, f32, #space>, #layout> *
        tensor<64x!cliff.multivector<1, unknown, f32, #space>, #layout> ->
        tensor<64x!cliff.multivector<105, unknown, f32, #space>, #layout>
        
    clg.store %pointer, %0 : !cliff.ptr<f32>, tensor<64x!cliff.multivector<105, unknown, f32, #space>, #layout>
    cliff.ret
}


cliff.func @complete_rotation(%x_ptr : !cliff.ptr<f32>, %y_ptr : !cliff.ptr<f32>, %angle_2_ptr : !cliff.ptr<f32>, %store_ptr : !cliff.ptr<f32>) {
    
    %x = clg.load %x_ptr : !cliff.ptr<f32> -> tensor<64x!cliff.multivector<105, unknown, f32, #space>, #layout>
    %y = clg.load %y_ptr : !cliff.ptr<f32> -> tensor<64x!cliff.multivector<105, unknown, f32, #space>, #layout>
    %angle_2 = clg.load %angle_2_ptr : !cliff.ptr<f32> -> tensor<64x!cliff.multivector<1, unknown, f32, #space>, #layout>
    
    %Z = cliff.rotate %x, %angle_2 :
        tensor<64x!cliff.multivector<105, unknown, f32, #space>, #layout> *
        tensor<64x!cliff.multivector<1, unknown, f32, #space>, #layout> ->
        tensor<64x!cliff.multivector<105, unknown, f32, #space>, #layout>

    %tmp = cliff.geo_prod %Z, %y :tensor<64x!cliff.multivector<105, unknown, f32, #space>, #layout> * tensor<64x!cliff.multivector<105, unknown, f32, #space>, #layout> -> tensor<64x!cliff.multivector<105, unknown, f32, #space>, #layout>
    %Z_rev = cliff.reverse %Z : tensor<64x!cliff.multivector<105, unknown, f32, #space>, #layout> -> tensor<64x!cliff.multivector<105, unknown, f32, #space>, #layout>
    %out = cliff.geo_prod %tmp, %Z_rev : tensor<64x!cliff.multivector<105, unknown, f32, #space>, #layout> * tensor<64x!cliff.multivector<105, unknown, f32, #space>, #layout> ->tensor<64x!cliff.multivector<105, unknown, f32, #space>, #layout>

    clg.store %store_ptr, %out : !cliff.ptr<f32>, tensor<64x!cliff.multivector<105, unknown, f32, #space>, #layout>
    cliff.ret
}