#space = #cliff.algebra<{p=3, q=0, r=1}>
module {

cliff.func @example(%arg0 : tensor<64x!cliff.multivector<3, 0000001, i32>>, %arg1 : tensor<64x!cliff.multivector<3, 0000001, i32>>) {

    %2 = cliff.geo_prod %arg0, %arg1 : tensor<64x!cliff.multivector<3, 1, i32>>
    cliff.ret %2 : tensor<64x!cliff.multivector<3, 0000001, i32>>

}

}