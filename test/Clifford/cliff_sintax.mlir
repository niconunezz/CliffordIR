#space = #cliff.algebra<{p=2, q=0, r=1}>

cliff.func @example(%arg0 : tensor<64x!cliff.multivector<20, f32, #space>>, 
                    %arg1 : tensor<64x!cliff.multivector<112, f32, #space>>, 
                    %time : tensor<64x32x!cliff.multivector<66, f32, #space>>) {
    
    %out = cliff.geo_prod %arg0, %time : 
        tensor<64x!cliff.multivector<20, f32, #space>> * 
        tensor<64x32x!cliff.multivector<66, f32, #space>> -> tensor<64x32x!cliff.multivector<60, f32, #space>>


    cliff.ret %out : tensor<64x32x!cliff.multivector<60, f32, #space>>
}
