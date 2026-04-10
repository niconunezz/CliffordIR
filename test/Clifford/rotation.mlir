#space = #cliff.algebra<{p=2, q=0, r=1}>


cliff.func @example(%arg0 : tensor<64x!cliff.point<euclidean, true, #space>>, 
                    %arg1 : tensor<64x!cliff.point<euclidean, true, #space>>, 
                    %time : tensor<64x32x!cliff.multivector<1, unknown, f32, #space>>) -> tensor<64x32x!cliff.point<euclidean, true, #space>> {
    
    %tmp = cliff.geo_prod %arg0, %time : 
        tensor<64x!cliff.point<euclidean, true, #space>> * 
        tensor<64x32x!cliff.multivector<1, unknown, f32, #space>> -> tensor<64x32x!cliff.point<euclidean, true, #space>>

    %motor = cliff.exp %tmp : tensor<64x32x!cliff.point<euclidean, true, #space>> -> tensor<64x32x!cliff.motor<true, #space>>


    %out = cliff.sandwich %motor, %arg1 : 
        tensor<64x32x!cliff.motor<true, #space>> *
        tensor<64x!cliff.point<euclidean, true, #space>> -> tensor<64x32x!cliff.point<euclidean, true, #space>>

    cliff.ret %out : tensor<64x32x!cliff.point<euclidean, true, #space>>
}
