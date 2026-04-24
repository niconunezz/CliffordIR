#space = #cliff.algebra<{p=2, q=0, r=1}>


cliff.func @example(%arg0 : !cliff.ptr<f32>, 
                    %arg1 : !cliff.ptr<f32>, 
                    %time : !cliff.ptr<f32>) -> tensor<64x32x!cliff.point<euclidean, true, #space>> {
    
    
    %t0 = cliff.load %arg0 : !cliff.ptr<f32> -> tensor<64x!cliff.point<euclidean, true, #space>>
    %t1 = cliff.load %arg1 : !cliff.ptr<f32> -> tensor<64x!cliff.point<euclidean, true, #space>>
    %t2 = cliff.load %time : !cliff.ptr<f32> -> tensor<64x32x!cliff.scalar<#space>>

    %tmp = cliff.geo_prod %t0, %t2 : 
        tensor<64x!cliff.point<euclidean, true, #space>> * 
        tensor<64x32x!cliff.scalar<#space>> -> tensor<64x32x!cliff.point<euclidean, true, #space>>

    %motor = cliff.exp %tmp : tensor<64x32x!cliff.point<euclidean, true, #space>> -> tensor<64x32x!cliff.motor<true, #space>>

    %out = cliff.sandwich %motor, %t1 : 
        tensor<64x32x!cliff.motor<true, #space>> *
        tensor<64x!cliff.point<euclidean, true, #space>> -> tensor<64x32x!cliff.point<euclidean, true, #space>>

    cliff.ret %out : tensor<64x32x!cliff.point<euclidean, true, #space>>
}
