module {

cliff.func @example(%arg0 : tensor<64x!cliff.multivector<3, 0000001, f64>>, %arg1 : tensor<64x!cliff.multivector<3, 0000001, f64>>) {

    %0 = cliff.geo_prod %arg0, %arg1 : tensor<64x!cliff.multivector<3, 1, f64>>

    cliff.ret %0 : tensor<64x!cliff.multivector<3, 0000001, f64>>

}

}