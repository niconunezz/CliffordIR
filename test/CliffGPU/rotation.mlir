#layout = #clg.linear<{register = [ [0]], lane = [ [1], [2], [4], [10], [16]], warp = [ [32], [64], [128]], block = [ [256], [512], [1024]]}>
#layout1 = #clg.linear<{register = [ [0, 0]], lane = [ [0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [ [0, 1], [0, 2], [0, 4]], block = [ [0, 32], [0, 64], [0, 128], [0, 256], [0, 512], [0, 1], [0, 2], [0, 4]]}>
#space = #cliff.algebra<{p=2, q=0, r=1}>
module {
  cliff.func @example(%arg0 : !cliff.ptr<f32>, 
                      %arg1 : !cliff.ptr<f32>, 
                      %time : !cliff.ptr<f32>) -> tensor<64x32x!cliff.point<euclidean, true, #space>> {
      
      %0 = clg.get_tid : i32
      
      %t0 = cliff.load %arg0 : !cliff.ptr<f32> -> tensor<64x!cliff.point<euclidean, true, #space>, #layout>
      %t1 = cliff.load %arg1 : !cliff.ptr<f32> -> tensor<64x!cliff.point<euclidean, true, #space>, #layout>
      %t2 = cliff.load %time : !cliff.ptr<f32> -> tensor<64x32x!cliff.scalar<#space>, #layout1>

      %tmp = cliff.geo_prod %t0, %t2 : 
          tensor<64x!cliff.point<euclidean, true, #space>, #layout> * 
          tensor<64x32x!cliff.scalar<#space>, #layout1> -> tensor<64x32x!cliff.point<euclidean, true, #space>, #layout1>

      %motor = cliff.exp %tmp : tensor<64x32x!cliff.point<euclidean, true, #space>, #layout1> -> tensor<64x32x!cliff.motor<true, #space>, #layout>

      %out = cliff.sandwich %motor, %t1 : 
          tensor<64x32x!cliff.motor<true, #space>, #layout1> *
          tensor<64x!cliff.point<euclidean, true, #space>, #layout> -> tensor<64x32x!cliff.point<euclidean, true, #space>, #layout1>

      cliff.ret %out : tensor<64x32x!cliff.point<euclidean, true, #space>, #layout1>
  }

}