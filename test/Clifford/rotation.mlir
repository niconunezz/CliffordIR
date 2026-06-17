#space = #cliff.algebra<{p=2, q=0, r=1}>
// RUN cliff-opt %s -split-input-file --geometric-type-conversion --convert-cliff-to-cliffGPU | FileCheck %s

// CHECK-LABEL: cliff.func @rotation
// CHECK-SAME: (%[[A0:.*]]: !clg.ptr<f32>, %[[A1:.*]]: !clg.ptr<f32>, %[[A2:.*]]: !clg.ptr<f32>, %[[OUT:.*]]: !clg.ptr<f32>)

// CHECK: clg.load %[[A0]]
// CHECK: clg.load %[[A1]]
// CHECK: clg.load %[[A2]]

// CHECK: cliff.geo_prod
// CHECK: cliff.exp
// CHECK: cliff.sandwich

// CHECK-NOT: cliff.ret %
// CHECK: clg.store %[[OUT]],
// CHECK: cliff.ret
// CHECK-NOT: cliff.ret %

cliff.func @rotation(%arg0 : tensor<64x!cliff.point<euclidean, true, #space>>,
                    %arg1 : tensor<64x!cliff.point<euclidean, true, #space>>,
                    %time : tensor<64x!cliff.scalar<#space>>,
                    %store : tensor<64x!cliff.point<euclidean, true, #space>>) -> tensor<64x!cliff.point<euclidean, true, #space>> {

    
    %tmp = cliff.geo_prod %arg0, %time : 
        tensor<64x!cliff.point<euclidean, true, #space>> * 
        tensor<64x!cliff.scalar<#space>> -> tensor<64x!cliff.point<euclidean, true, #space>>

    %motor = cliff.exp %tmp : tensor<64x!cliff.point<euclidean, true, #space>> -> tensor<64x!cliff.motor<true, #space>>

    %out = cliff.sandwich %motor, %arg1 : 
        tensor<64x!cliff.motor<true, #space>> *
        tensor<64x!cliff.point<euclidean, true, #space>> -> tensor<64x!cliff.point<euclidean, true, #space>>

    cliff.ret %out : tensor<64x!cliff.point<euclidean, true, #space>>
}