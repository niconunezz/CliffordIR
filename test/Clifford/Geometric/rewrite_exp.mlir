// RUN: cliff-opt %s -split-input-file --rewrite-exponential | FileCheck %s 

// -----

// CHECK-LABEL: cliff.func @rotation2D
// CHECK:   cliff.geo_prod
// CHECK:       cliff.rotate
// CHECK-NOT:   cliff.exp
#space = #cliff.algebra<{p=2, q=0, r=1}>
cliff.func @rotation2D(%arg0 : tensor<64x!cliff.point<euclidean, true, #space>>,
                    %time : tensor<64x32x!cliff.scalar<#space>>) -> tensor<64x32x!cliff.motor<true, #space>> {
    
    %tmp = cliff.geo_prod %time, %arg0 : tensor<64x32x!cliff.scalar<#space>> * 
        tensor<64x!cliff.point<euclidean, true, #space>> -> tensor<64x32x!cliff.point<euclidean, true, #space>>

    %motor = cliff.exp %tmp : tensor<64x32x!cliff.point<euclidean, true, #space>> -> tensor<64x32x!cliff.motor<true, #space>>

    cliff.ret %motor : tensor<64x32x!cliff.motor<true, #space>>
}

// -----

// CHECK-LABEL: cliff.func @translation2D
// CHECK:   cliff.geo_prod
// CHECK:       cliff.translate
// CHECK-NOT:   cliff.exp
#space = #cliff.algebra<{p=2, q=0, r=1}>
cliff.func @translation2D(%arg0 : tensor<64x!cliff.point<ideal, true, #space>>, 
                    %time : tensor<64x32x!cliff.scalar<#space>>) -> tensor<64x32x!cliff.motor<true, #space>> {
    
    %tmp = cliff.geo_prod %time, %arg0 : tensor<64x32x!cliff.scalar<#space>> * 
        tensor<64x!cliff.point<ideal, true, #space>> -> tensor<64x32x!cliff.point<ideal, true, #space>>

    %motor = cliff.exp %tmp : tensor<64x32x!cliff.point<ideal, true, #space>> -> tensor<64x32x!cliff.motor<true, #space>>

    cliff.ret %motor : tensor<64x32x!cliff.motor<true, #space>>
}

// -----

// CHECK-LABEL: cliff.func @rotation3D
// CHECK:   cliff.geo_prod
// CHECK:       cliff.rotate
// CHECK-NOT:   cliff.exp
#space = #cliff.algebra<{p=3, q=0, r=1}>
cliff.func @rotation3D(%arg0 : tensor<64x!cliff.line<euclidean, true, #space>>, 
                    %time : tensor<64x32x!cliff.scalar<#space>>) -> tensor<64x32x!cliff.motor<true, #space>> {
    
    %tmp = cliff.geo_prod %time, %arg0 : tensor<64x32x!cliff.scalar<#space>> * 
        tensor<64x!cliff.line<euclidean, true, #space>> -> tensor<64x!cliff.line<euclidean, true, #space>>

    %motor = cliff.exp %tmp : tensor<64x!cliff.line<euclidean, true, #space>> -> tensor<64x32x!cliff.motor<true, #space>>

    cliff.ret %motor : tensor<64x32x!cliff.motor<true, #space>>
}

// -----

// CHECK-LABEL: cliff.func @translation3D
// CHECK:   cliff.geo_prod
// CHECK:       cliff.translate
// CHECK-NOT:   cliff.exp
#space = #cliff.algebra<{p=3, q=0, r=1}>
cliff.func @translation3D(%arg0 : tensor<64x!cliff.line<ideal, true, #space>>, 
                    %time : tensor<64x32x!cliff.scalar<#space>>) -> tensor<64x32x!cliff.motor<true, #space>> {
    
    %tmp = cliff.geo_prod %time, %arg0 : tensor<64x32x!cliff.scalar<#space>> * 
        tensor<64x!cliff.line<ideal, true, #space>> -> tensor<64x!cliff.line<ideal, true, #space>>

    %motor = cliff.exp %tmp : tensor<64x!cliff.line<ideal, true, #space>> -> tensor<64x32x!cliff.motor<true, #space>>

    cliff.ret %motor : tensor<64x32x!cliff.motor<true, #space>>
}