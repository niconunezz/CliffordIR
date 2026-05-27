// RUN: cliff-opt %s -split-input-file --rewrite-sandwich | FileCheck %s 

#space = #cliff.algebra<{p=2, q=0, r=1}>


// CHECK-LABEL: cliff.func @rewrite_sandwich
// CHECK-NOT:   cliff.sandwich
// CHECK:       cliff.geo_prod
// CHECK:   cliff.reverse
// CHECK:       cliff.geo_prod

cliff.func @rewrite_sandwich(%arg0 : tensor<64x!cliff.point<euclidean, true, #space>>, 
                    %arg1 : tensor<64x!cliff.point<euclidean, true, #space>>, 
                    %time : tensor<64x32x!cliff.scalar<#space>>) -> tensor<64x32x!cliff.point<euclidean, true, #space>> {
    
    %tmp = cliff.geo_prod %time, %arg0 :tensor<64x32x!cliff.scalar<#space>> * 
        tensor<64x!cliff.point<euclidean, true, #space>> -> tensor<64x32x!cliff.point<euclidean, true, #space>>

    %motor = cliff.exp %tmp : tensor<64x32x!cliff.point<euclidean, true, #space>> -> tensor<64x32x!cliff.motor<true, #space>>
    %out = cliff.sandwich %motor, %arg1 : 
        tensor<64x32x!cliff.motor<true, #space>> *
        tensor<64x!cliff.point<euclidean, true, #space>> -> tensor<64x32x!cliff.point<euclidean, true, #space>>

    cliff.ret %out : tensor<64x32x!cliff.point<euclidean, true, #space>>
}
