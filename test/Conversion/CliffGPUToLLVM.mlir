// RUN: cliff-opt %s -split-input-file --convert-cliffGPU-to-llvm | FileCheck %s

#space = #cliff.algebra<{p=2, q=0, r=1}>
#layout = #clg.linear<{register = [[0]], lane = [[1],[2],[4],[8],[16]], warp = [[32],[64],[128]], block = [[256],[512],[1024]]}>
#layout1 = #clg.linear<{register = [[0,0]], lane = [[0,1],[0,2],[0,4],[0,8],[0,16]], warp = [[0,1],[0,2],[0,4]], block = [[0,32],[0,64],[0,128],[0,256],[0,512],[0,1],[0,2],[0,4]]}>

// CHECK-LABEL: llvm.func @geo_prod_scalar_case
// CHECK-SAME: (%[[ARG0:.*]]: !llvm.struct<(struct(f32, f32, f32))>, %[[ARG1:.*]]: !llvm.struct<(struct(f32))>)
// CHECK-SAME: -> !llvm.struct<(struct(f32, f32, f32))>
cliff.func @geo_prod_scalar_case(
    %arg0: tensor<64x!cliff.multivector<11, unknown, f32, #space>, #layout>,
    %arg1: tensor<64x32x!cliff.multivector<1, unknown, f32, #space>, #layout1>) -> tensor<64x32x!cliff.multivector<11, unknown, f32, #space>, #layout1> {

    // CHECK: %[[A0:.*]] = llvm.extractvalue %[[ARG0]][0] : !llvm.struct<(struct(f32, f32, f32))>
    // CHECK: %[[A1:.*]] = llvm.extractvalue %[[ARG1]][0] : !llvm.struct<(struct(f32))>

    // CHECK: %[[OUT:.*]] = llvm.mlir.undef : !llvm.struct<(struct(f32, f32, f32))>

    // CHECK: %[[S:.*]] = llvm.extractvalue %[[A1]][0] : !llvm.struct(f32)

    // CHECK: %[[L0:.*]] = llvm.extractvalue %[[A0]][0] : !llvm.struct(f32, f32, f32)
    // CHECK: %[[M0:.*]] = arith.mulf %[[S]], %[[L0]] : f32
    // CHECK: %[[R0:.*]] = llvm.insertvalue %[[M0]], %[[OUT]][0, 0] : !llvm.struct<(struct(f32, f32, f32))>

    // CHECK: %[[L1:.*]] = llvm.extractvalue %[[A0]][1] : !llvm.struct(f32, f32, f32)
    // CHECK: %[[M1:.*]] = arith.mulf %[[S]], %[[L1]] : f32
    // CHECK: %[[R1:.*]] = llvm.insertvalue %[[M1]], %[[R0]][0, 1] : !llvm.struct<(struct(f32, f32, f32))>

    // CHECK: %[[L2:.*]] = llvm.extractvalue %[[A0]][2] : !llvm.struct(f32, f32, f32)
    // CHECK: %[[M2:.*]] = arith.mulf %[[S]], %[[L2]] : f32
    // CHECK: %[[R2:.*]] = llvm.insertvalue %[[M2]], %[[R1]][0, 2] : !llvm.struct<(struct(f32, f32, f32))>

    // CHECK: llvm.return %[[R2]] : !llvm.struct<(struct(f32, f32, f32))>

    %0 = cliff.geo_prod %arg0, %arg1 :
        tensor<64x!cliff.multivector<11, unknown, f32, #space>, #layout> *
        tensor<64x32x!cliff.multivector<1, unknown, f32, #space>, #layout1> ->
        tensor<64x32x!cliff.multivector<11, unknown, f32, #space>, #layout1>
    cliff.ret %0 : tensor<64x32x!cliff.multivector<11, unknown, f32, #space>, #layout1>
}

// -----

#layout = #clg.linear<{register = [ [0]], lane = [ [1], [2], [4], [10], [16]], warp = [ [32], [64], [128]], block = [ [256], [512], [1024]]}>
#layout1 = #clg.linear<{register = [ [0, 0]], lane = [ [0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [ [0, 1], [0, 2], [0, 4]], block = [ [0, 32], [0, 64], [0, 128], [0, 256], [0, 512], [0, 1], [0, 2], [0, 4]]}>
#space = #cliff.algebra<{p = 2, q =0, r = 1}>

// CHECK-LABEL: llvm.func @geo_prod_general_case(
// CHECK-SAME:    %[[A0:[a-zA-Z0-9_]+]]: !llvm.struct<(struct(f32, f32, f32))>,
// CHECK-SAME:    %[[A1:[a-zA-Z0-9_]+]]: !llvm.struct<(struct(f32, f32, f32))>)
// CHECK-SAME:    -> !llvm.struct<(struct(f32, f32, f32, f32))>

cliff.func @geo_prod_general_case(%arg0: tensor<64x!cliff.multivector<11, unknown, f32, #space>, #layout>, %arg1: tensor<64x32x!cliff.multivector<7, unknown, f32, #space>, #layout1>) -> tensor<64x32x!cliff.multivector<15, unknown, f32, #space>, #layout1>{

// CHECK:    %[[ARR0:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[A0]][0] : !llvm.struct<(struct(f32, f32, f32))>
// CHECK:    %[[ARR1:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[A1]][0] : !llvm.struct<(struct(f32, f32, f32))>

// CHECK:    %[[RES:[a-zA-Z0-9_]+]] = llvm.mlir.undef : !llvm.struct<(struct(f32, f32, f32, f32))>

// CHECK:    %[[E00:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[ARR0]][0] : !llvm.struct(f32, f32, f32)
// CHECK:    %[[E10:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[ARR1]][0] : !llvm.struct(f32, f32, f32)
// CHECK:    %[[MUL0:[a-zA-Z0-9_]+]] = arith.mulf %[[E00]], %[[E10]] : f32
// CHECK:    {{.*}} = arith.addf {{.*}}, %[[MUL0]] : f32
// CHECK:    {{.*}} = llvm.insertvalue {{.*}}[0, 0] : !llvm.struct<(struct(f32, f32, f32, f32))>

// CHECK:    {{.*}} = llvm.extractvalue %[[ARR1]][1] : !llvm.struct(f32, f32, f32)
// CHECK:    {{.*}} = arith.mulf %[[E00]], {{.*}} : f32
// CHECK:    {{.*}} = arith.addf {{.*}}, {{.*}} : f32
// CHECK:    {{.*}} = llvm.insertvalue {{.*}}[0, 1] : !llvm.struct<(struct(f32, f32, f32, f32))>

// CHECK:    {{.*}} = llvm.extractvalue %[[ARR1]][2] : !llvm.struct(f32, f32, f32)
// CHECK:    {{.*}} = arith.mulf %[[E00]], {{.*}} : f32
// CHECK:    {{.*}} = arith.addf {{.*}}, {{.*}} : f32
// CHECK:    {{.*}} = llvm.insertvalue {{.*}}[0, 2] : !llvm.struct<(struct(f32, f32, f32, f32))>

// CHECK:    %[[E01:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[ARR0]][1] : !llvm.struct(f32, f32, f32)
// CHECK:    {{.*}} = arith.mulf %[[E01]], {{.*}} : f32
// CHECK:    {{.*}} = llvm.insertvalue {{.*}}[0, 1] : !llvm.struct<(struct(f32, f32, f32, f32))>
// CHECK:    {{.*}} = arith.mulf %[[E01]], {{.*}} : f32
// CHECK:    {{.*}} = llvm.insertvalue {{.*}}[0, 3] : !llvm.struct<(struct(f32, f32, f32, f32))>

// CHECK:    %[[E02:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[ARR0]][2] : !llvm.struct(f32, f32, f32)
// CHECK:    {{.*}} = arith.mulf %[[E02]], {{.*}} : f32
// CHECK:    {{.*}} = llvm.insertvalue {{.*}}[0, 3] : !llvm.struct<(struct(f32, f32, f32, f32))>
// CHECK:    {{.*}} = arith.mulf %[[E02]], {{.*}} : f32
// CHECK:    {{.*}} = llvm.insertvalue {{.*}}[0, 1] : !llvm.struct<(struct(f32, f32, f32, f32))>

// CHECK:    llvm.return {{.*}} : !llvm.struct<(struct(f32, f32, f32, f32))>

    %0 = cliff.geo_prod %arg0, %arg1 : tensor<64x!cliff.multivector<11, unknown, f32, #space>, #layout> * tensor<64x32x!cliff.multivector<7, unknown, f32, #space>, #layout1> -> tensor<64x32x!cliff.multivector<15, unknown, f32, #space>, #layout1>
    cliff.ret %0 : tensor<64x32x!cliff.multivector<15, unknown, f32, #space>, #layout1>
}

// -----
#space = #cliff.algebra<{p=2, q=0, r=1}>
#layout = #clg.linear<{register = [[0]], lane = [[1],[2],[4],[8],[16]], warp = [[32],[64],[128]], block = [[256],[512],[1024]]}>
// CHECK-LABEL: llvm.func @reverse_to_llvm
// CHECK: %[[OUTER:.+]] = llvm.extractvalue %{{.*}}[0]
// CHECK: %[[C0:.+]] = llvm.extractvalue %[[OUTER]][0]
// CHECK: %[[C1:.+]] = llvm.extractvalue %[[OUTER]][1]
// CHECK: %[[C2:.+]] = llvm.extractvalue %[[OUTER]][2]
// CHECK: %[[C3:.+]] = llvm.extractvalue %[[OUTER]][3]
// CHECK: %[[C4:.+]] = llvm.extractvalue %[[OUTER]][4]
// CHECK: %[[C5:.+]] = llvm.extractvalue %[[OUTER]][5]
// CHECK: %[[C6:.+]] = llvm.extractvalue %[[OUTER]][6]
// CHECK: %[[NEG3:.+]] = llvm.fneg %[[C3]]
// CHECK: %[[NEG5:.+]] = llvm.fneg %[[C5]]
// CHECK: %[[NEG6:.+]] = llvm.fneg %[[C6]]
// CHECK: llvm.insertvalue %[[C0]], %{{.*}}[0]
// CHECK: llvm.insertvalue %[[C1]], %{{.*}}[1]
// CHECK: llvm.insertvalue %[[C2]], %{{.*}}[2]
// CHECK: llvm.insertvalue %[[NEG3]], %{{.*}}[3]
// CHECK: llvm.insertvalue %[[C4]], %{{.*}}[4]
// CHECK: llvm.insertvalue %[[NEG5]], %{{.*}}[5]
// CHECK: llvm.insertvalue %[[NEG6]], %{{.*}}[6]

cliff.func @reverse_to_llvm(%arg0: tensor<64x!cliff.multivector<127, unknown, f32, #space>, #layout>) -> tensor<64x!cliff.multivector<127, unknown, f32, #space>, #layout> {
    %0 = cliff.reverse %arg0 : tensor<64x!cliff.multivector<127, unknown, f32, #space>, #layout> ->
                               tensor<64x!cliff.multivector<127, unknown, f32, #space>, #layout>
    
    cliff.ret %0 : tensor<64x!cliff.multivector<127, unknown, f32, #space>, #layout>
}

