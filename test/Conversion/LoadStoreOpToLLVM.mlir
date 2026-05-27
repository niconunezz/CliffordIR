// RUN: cliff-opt %s -split-input-file --convert-cliffGPU-to-llvm | FileCheck %s

#layout = #clg.linear<{register = [], lane = [ [1], [2], [4], [8], [16]], warp = [[32]], block = []}>

#space = #cliff.algebra<{p=2, q=0, r=1}>
module {
  // CHECK-LABEL: llvm.func @load1D
  cliff.func @load1D(%arg0 : !cliff.ptr<f32>) -> tensor<64x!cliff.multivector<104, point, f32, #space>, #layout> {
    // CHECK: nvvm.read.ptx.sreg.tid.x
   
    // CHECK: llvm.mul {{.*}}, %{{[0-9]+}} : i32   
    // CHECK-COUNT-3: llvm.load %{{.*}} : !llvm.ptr -> f32
    // CHECK-COUNT-3: llvm.insertvalue
    %t0 = clg.load %arg0 : !cliff.ptr<f32> -> tensor<64x!cliff.multivector<104, point, f32, #space>, #layout>
    cliff.ret %t0 : tensor<64x!cliff.multivector<104, point, f32, #space>, #layout>
  
  }
}

// // -----

#layout = #clg.linear<{register = [], lane = [ [1], [2], [4], [8], [16]], warp = [[32]], block = []}>

#space = #cliff.algebra<{p=2, q=0, r=1}>
module {
  // CHECK-LABEL: llvm.func @store1D
  cliff.func @store1D(%arg0 : tensor<64x!cliff.multivector<104, point, f32, #space>, #layout>, %pointer : !cliff.ptr<f32>) {

    clg.store %pointer, %arg0 : !cliff.ptr<f32>, tensor<64x!cliff.multivector<104, point, f32, #space>, #layout>
    cliff.ret
  }
}
