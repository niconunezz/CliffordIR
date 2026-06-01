#layout = #clg.linear<{register = [ [0]], lane = [ [1], [2], [4], [10], [16]], warp = [ [32], [64], [128]], block = [ [256], [512], [1024]]}>
#space = #cliff.algebra<{p=2, q=0, r=1}>
module {
  cliff.func @example(%arg0 : !cliff.ptr<f32>, 
                      %arg1 : !cliff.ptr<f32>, 
                      %time : !cliff.ptr<f32>) -> tensor<64x!cliff.point<euclidean, true, #space>> {
              
      
      %t0 = cliff.load %arg0 : !cliff.ptr<f32> -> tensor<64x!cliff.point<euclidean, true, #space>, #layout>
      %t1 = cliff.load %arg1 : !cliff.ptr<f32> -> tensor<64x!cliff.point<euclidean, true, #space>, #layout>
      %t2 = cliff.load %time : !cliff.ptr<f32> -> tensor<64x!cliff.scalar<#space>, #layout>

      %tmp = cliff.geo_prod %t0, %t2 : 
          tensor<64x!cliff.point<euclidean, true, #space>, #layout> * 
          tensor<64x!cliff.scalar<#space>, #layout> -> tensor<64x!cliff.point<euclidean, true, #space>, #layout>

      %motor = cliff.exp %tmp : tensor<64x!cliff.point<euclidean, true, #space>, #layout> -> tensor<64x!cliff.motor<true, #space>, #layout>

      %out = cliff.sandwich %motor, %t1 : 
          tensor<64x!cliff.motor<true, #space>, #layout> *
          tensor<64x!cliff.point<euclidean, true, #space>, #layout> -> tensor<64x!cliff.point<euclidean, true, #space>, #layout>
          

      cliff.ret %out : tensor<64x!cliff.point<euclidean, true, #space>, #layout>
  }

}