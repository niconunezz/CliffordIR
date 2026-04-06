#space = #cliff.algebra<{p=2, q=0, r=1}>


cliff.func @example(%arg0 : tensor<64x!cliff.multivector<112, f32, #space>>, 
                    %arg1 : tensor<64x!cliff.multivector<112, f32, #space>>, 
                    %time : tensor<64x32x!cliff.multivector<0, f32, #space>>) {
    
    %tmp = cliff.geo_prod %arg0, %time : 
        tensor<64x!cliff.multivector<112, f32, #space>> * 
        tensor<64x32x!cliff.multivector<0, f32, #space>> -> tensor<64x32x!cliff.multivector<112, f32, #space>>

    %motor = cliff.exp %tmp : tensor<64x32x!cliff.multivector<112, f32, #space>>


    %out = cliff.sandwich %motor, %arg1 : 
        tensor<64x32x!cliff.multivector<112, f32, #space>> *
        tensor<64x!cliff.multivector<112, f32, #space>> -> tensor<64x32x!cliff.multivector<112, f32, #space>>

    cliff.ret %out : tensor<64x32x!cliff.multivector<112, f32, #space>>
}
