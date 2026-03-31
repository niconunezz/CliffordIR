#space = #cliff.algebra<{p=2, q=0, r=1}>
#global = #clg.linear<{register = [[0]], lane = [[1], [2], [4], [8], [16]], warp = [[0]], block = [[0]]}>

cliff.func @example(%arg0 : tensor<1x!cliff.multivector<01110000, f32, #space>, #global>, 
                    %arg1 : tensor<1x!cliff.multivector<01110000, f32, #space>, #global>, 
                    %time : tensor<1x32x!cliff.multivector<00000001, f32, #space>, #global>) {
    
    %tmp = cliff.geo_prod %arg0, %time : 
        tensor<1x!cliff.multivector<01110000, f32, #space>, #global> * 
        tensor<1x32x!cliff.multivector<00000001, f32, #space>, #global> -> 
        tensor<1x32x!cliff.multivector<01110000, f32, #space>, #global>

    %motor = cliff.exp %tmp : tensor<1x32x!cliff.multivector<01110000, f32, #space>, #global>


    %out = cliff.sandwich %motor, %arg1 : 
        tensor<1x32x!cliff.multivector<01110000, f32, #space>, #global> *
        tensor<1x!cliff.multivector<01110000, f32, #space>, #global> -> tensor<1x32x!cliff.multivector<01110000, f32, #space>, #global>

    cliff.ret %out : tensor<1x32x!cliff.multivector<01110000, f32, #space>, #global>
}
