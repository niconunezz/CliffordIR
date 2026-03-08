#space = #cliff.algebra<{p=2, q=0, r=1}>


cliff.func @example(%arg0 : tensor<64x!cliff.multivector<01110000, f32, #space>>, 
                    %arg1 : tensor<64x!cliff.multivector<01110000, f32, #space>>, 
                    %angle : tensor<64x!cliff.multivector<00000001, f32, #space>>,
                    %time : tensor<64x!cliff.multivector<00000001, f32, #space>>) {

    %0 = arith.constant 0
    %ones = arith.constant dense<1.0> : tensor<64xf32>

    %xy = cliff.geo_prod %arg0, %arg1 : 
        tensor<64x!cliff.multivector<01110000, f32, #space>> * 
        tensor<64x!cliff.multivector<01110000, f32, #space>> -> tensor<64x!cliff.multivector<01110000, f32, #space>>

    // sqrt
    %scalar = cliff.get_idx %xy, %0 : 
        tensor<64x!cliff.multivector<01110000, f32, #space>> -> tensor<64x!cliff.multivector<00000001, f32, #space>>

    %sign = cliff.sign %scalar : tensor<64x!cliff.multivector<00000001, f32, #space>>
    %sum = cliff.add %xy, %sign : tensor<64x!cliff.multivector<01110001, f32, #space>>
    %norm = cliff.normalize %sum : tensor<64x!cliff.multivector<01110001, f32, #space>>
    // lerp
    %scalar1 = cliff.get_idx %norm, %0 : 
        tensor<64x!cliff.multivector<01110000, f32, #space>> -> tensor<64x!cliff.multivector<00000001, f32, #space>>

    %sign2 = cliff.sign %scalar1 : tensor<64x!cliff.multivector<00000001, f32, #space>>

    %neg = 
    %factor2 = cliff.add %time, %ones : 
    
    %tmp = cliff.geo_prod %arg0, %angle : 
        tensor<64x!cliff.multivector<01110000, f32, #space>> * 
        tensor<64x!cliff.multivector<00000001, f32, #space>> -> tensor<64x!cliff.multivector<01110000, f32, #space>>

    %motor = cliff.exp %tmp : tensor<64x!cliff.multivector<01110000, f32, #space>>


    cliff.ret %2 : tensor<64x!cliff.multivector<01110000, f32, #space>>
}
