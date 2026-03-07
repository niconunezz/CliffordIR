#space = #cliff.algebra<{p=3, q=0, r=1}>
module {

cliff.func @example(%arg0 : tensor<64x!cliff.multivector<000000000000001, i32, #space>>, %arg1 : tensor<64x!cliff.multivector<000000000000001, i32, #space>>) {

    %2 = cliff.geo_prod %arg0, %arg1 : 
        tensor<64x!cliff.multivector<000000000000001, i32, #space>> * 
        tensor<64x!cliff.multivector<000000000000001, i32, #space>> -> tensor<64x!cliff.multivector<000000000000000, i32, #space>>
    cliff.ret %2 : tensor<64x!cliff.multivector<000000000000000, i32, #space>>

}

}