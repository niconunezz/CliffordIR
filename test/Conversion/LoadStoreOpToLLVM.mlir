// RUN: cliff-opt %s -split-input-file --convert-cliffGPU-to-llvm | FileCheck %s

#layout = #clg.linear<{register = [], lane = [ [1], [2], [4], [8], [16]], warp = [[32]], block = []}>

#space = #cliff.algebra<{p=2, q=0, r=1}>
module {
  // CHECK-LABEL: llvm.func @load1D
  cliff.func @load1D(%arg0 : !clg.ptr<f32>) -> tensor<64x!cliff.multivector<104, point, f32, #space>, #layout> {
    // CHECK: nvvm.read.ptx.sreg.tid.x
   
    // CHECK: llvm.mul {{.*}}, %{{[0-9]+}} : i32   
    // CHECK-COUNT-3: llvm.load %{{.*}} : !llvm.ptr -> f32
    // CHECK-COUNT-3: llvm.insertvalue
    %t0 = clg.load %arg0 : !clg.ptr<f32> -> tensor<64x!cliff.multivector<104, point, f32, #space>, #layout>
    cliff.ret %t0 : tensor<64x!cliff.multivector<104, point, f32, #space>, #layout>
  
  }
}

// -----

#layout = #clg.linear<{register = [], lane = [ [1], [2], [4], [8], [16]], warp = [[32]], block = []}>

#space = #cliff.algebra<{p=2, q=0, r=1}>
module {
  // CHECK-LABEL: llvm.func @store1D
  // CHECK-COUNT-3: llvm.extractvalue
  // CHECK-COUNT-3: llvm.store

  cliff.func @store1D(%arg0 : tensor<64x!cliff.multivector<104, point, f32, #space>, #layout>, %pointer : !clg.ptr<f32>) {

    clg.store %pointer, %arg0 : !clg.ptr<f32>, tensor<64x!cliff.multivector<104, point, f32, #space>, #layout>
    cliff.ret
  }
}

// -----

#layout = #clg.linear<{register = [[1]], lane = [ [1], [2], [4], [8], [16]], warp = [[32]], block = []}>

#space = #cliff.algebra<{p=2, q=0, r=1}>
module {
  // CHECK-LABEL: llvm.func @load2Regs
  cliff.func @load2Regs(%arg0 : !clg.ptr<f32>) -> tensor<64x!cliff.multivector<104, point, f32, #space>, #layout> {
    // CHECK: nvvm.read.ptx.sreg.tid.x
   
    // CHECK: llvm.mul {{.*}}, %{{[0-9]+}} : i32   
    // CHECK-COUNT-3: llvm.load %{{.*}} : !llvm.ptr -> f32
    // CHECK-COUNT-3: llvm.insertvalue
    // CHECK-COUNT-3: llvm.load %{{.*}} : !llvm.ptr -> f32
    // CHECK-COUNT-3: llvm.insertvalue
    // CHECK-COUNT-2: llvm.insertvalue


    %t0 = clg.load %arg0 : !clg.ptr<f32> -> tensor<64x!cliff.multivector<104, point, f32, #space>, #layout>
    cliff.ret %t0 : tensor<64x!cliff.multivector<104, point, f32, #space>, #layout>
  
  }
}


// -----

#layout = #clg.linear<{register = [[1]], lane = [ [1], [2], [4], [8], [16]], warp = [[32]], block = []}>

#space = #cliff.algebra<{p=2, q=0, r=1}>
module {
  // CHECK-LABEL: llvm.func @store2regs
  // CHECK: %[[V14:.*]] = llvm.extractvalue %arg0[0] : !llvm.struct<(struct<(f32, f32, f32)>, struct<(f32, f32, f32)>)>
  // CHECK-NEXT: %[[V15:.*]] = llvm.extractvalue %arg0[1] : !llvm.struct<(struct<(f32, f32, f32)>, struct<(f32, f32, f32)>)>
  // CHECK-NEXT: %[[V16:.*]] = llvm.extractvalue %[[V14]][0] : !llvm.struct<(f32, f32, f32)>
  // CHECK-NEXT: %[[V17:.*]] = llvm.extractvalue %[[V14]][1] : !llvm.struct<(f32, f32, f32)>
  // CHECK-NEXT: %[[V18:.*]] = llvm.extractvalue %[[V14]][2] : !llvm.struct<(f32, f32, f32)>

  // CHECK-COUNT-6: llvm.store
  cliff.func @store2regs(%arg0 : tensor<64x!cliff.multivector<104, point, f32, #space>, #layout>, %pointer : !clg.ptr<f32>) {

    clg.store %pointer, %arg0 : !clg.ptr<f32>, tensor<64x!cliff.multivector<104, point, f32, #space>, #layout>
    cliff.ret
  }
}
