#layout = #clg.linear<{register = [ [0]], lane = [ [1], [2], [4], [10], [16]], warp = [ [32], [64], [128]], block = [ [256], [512], [1024]]}>
#layout1 = #clg.linear<{register = [ [0, 0]], lane = [ [0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [ [0, 1], [0, 2], [0, 4]], block = [ [0, 32], [0, 64], [0, 128], [0, 256], [0, 512], [0, 1], [0, 2], [0, 4]]}>
#space = #cliff.algebra<{p = 2, q =0, r = 1}>
module {
  cliff.func @example(%arg0: tensor<64x!cliff.multivector<10, f32, #space>, #layout>, %arg1: tensor<64x!cliff.multivector<10, f32, #space>, #layout>, %arg2: tensor<64x32x!cliff.multivector<0, f32, #space>, #layout1>) -> tensor<64x32x!cliff.multivector<10, f32, #space>, #layout1>{
    // %1 = cliff.exp %0 : tensor<64x32x!cliff.multivector<10, f32, #space>, #layout1>
    // %2 = cliff.sandwich %1, %arg1 : tensor<64x32x!cliff.multivector<10, f32, #space>, #layout1> * tensor<64x!cliff.multivector<10, f32, #space>, #layout> -> tensor<64x32x!cliff.multivector<10, f32, #space>, #layout1>
    %0 = cliff.geo_prod %arg0, %arg2 : tensor<64x!cliff.multivector<10, f32, #space>, #layout> * tensor<64x32x!cliff.multivector<0, f32, #space>, #layout1> -> tensor<64x32x!cliff.multivector<10, f32, #space>, #layout1>

    cliff.ret %0 : tensor<64x32x!cliff.multivector<10, f32, #space>, #layout1>
  }
}

