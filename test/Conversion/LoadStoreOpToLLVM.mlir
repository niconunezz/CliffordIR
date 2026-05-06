// RUN: cliff-opt %s -split-input-file --convert-cliffGPU-to-llvm | FileCheck %s

#layout = #clg.linear<{register = [ [0]], lane = [ [1], [2], [4], [10], [16]], warp = [ [32], [64], [128]], block = [ [256], [512], [1024]]}>
#layout1 = #clg.linear<{register = [ [0, 0]], lane = [ [0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [ [0, 1], [0, 2], [0, 4]], block = [ [0, 32], [0, 64], [0, 128], [0, 256], [0, 512], [0, 1], [0, 2], [0, 4]]}>
#space = #cliff.algebra<{p=2, q=0, r=1}>
module {
  cliff.func @example(%arg0 : !cliff.ptr<f32>, 
                      %arg1 : !cliff.ptr<f32>, 
                      %time : !cliff.ptr<f32>) -> tensor<64x!cliff.multivector<15, point, f32, #space>, #layout> {
              
      
      %t0 = clg.load %arg0 : !cliff.ptr<f32> -> tensor<64x!cliff.multivector<15, point, f32, #space>, #layout>
    //   %t2 = cliff.load %time : !cliff.ptr<f32> -> tensor<64x32x!cliff.scalar<#space>, #layout1>
      cliff.ret %t0 : tensor<64x!cliff.multivector<15, point, f32, #space>, #layout>
  }

}

// --geometric-type-conversion