#space = #cliff.algebra<{p=2, q=0, r=1}>


cliff.func @example(%arg0 : tensor<64x!cliff.multivector<01110000, f32, #space>>, 
                    %arg1 : tensor<64x!cliff.multivector<01110000, f32, #space>>, 
                    %time : tensor<64x!cliff.multivector<00000001, f32, #space>>, 
                    %time2 : tensor<64x!cliff.multivector<00000001, f32, #space>>) {

    %0 = arith.constant 0 : i32

    %ones = cliff.constant dense<1> : tensor<64x!cliff.multivector<00000001, f32, #space>>

    %xy = cliff.geo_prod %arg0, %arg1 : 
        tensor<64x!cliff.multivector<01110000, f32, #space>> * 
        tensor<64x!cliff.multivector<01110000, f32, #space>> -> tensor<64x!cliff.multivector<01110001, f32, #space>>

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

    %factor2 = cliff.sub %ones, %time : tensor<64x!cliff.multivector<00000001, f32, #space>>

    %prod1 = cliff.geo_prod %sign2, %factor2 : 
        tensor<64x!cliff.multivector<00000001, f32, #space>> *
        tensor<64x!cliff.multivector<00000001, f32, #space>> -> tensor<64x!cliff.multivector<00000001, f32, #space>>

    
    %prod2 = cliff.geo_prod %time, %arg0 : 
        tensor<64x!cliff.multivector<00000001, f32, #space>> *
        tensor<64x!cliff.multivector<01110000, f32, #space>> -> tensor<64x!cliff.multivector<01110000, f32, #space>>

    %pre_norm = cliff.add %prod1, %prod2 : 
        tensor<64x!cliff.multivector<01110000, f32, #space>> +
        tensor<64x!cliff.multivector<00000001, f32, #space>> -> tensor<64x!cliff.multivector<01110001, f32, #space>>

    %U = cliff.normalize %pre_norm : tensor<64x!cliff.multivector<01110001, f32, #space>>
    
    %tmp = cliff.geo_prod %arg0, %time2 : 
        tensor<64x!cliff.multivector<01110000, f32, #space>> * 
        tensor<64x!cliff.multivector<00000001, f32, #space>> -> tensor<64x!cliff.multivector<01110000, f32, #space>>

    %motor = cliff.exp %tmp : tensor<64x!cliff.multivector<01110000, f32, #space>>


    cliff.ret %2 : tensor<64x!cliff.multivector<01110000, f32, #space>>
}
