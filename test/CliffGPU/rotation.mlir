#layout = #clg.linear<{register = [], lane = [ [1], [2], [4], [8], [16]], warp = [[32]], block = []}>
#space = #cliff.algebra<{p=2, q=0, r=1}>

module {
  cliff.func @example(%arg0 : !cliff.ptr<f32>, 
                      %arg1 : !cliff.ptr<f32>, 
                      %y : !cliff.ptr<f32>,
                      %pointer : !cliff.ptr<f32>) {
              
      
      %t0 = clg.load %arg0 : !cliff.ptr<f32> -> tensor<64x!cliff.multivector<105, unknown, f32, #space>, #layout>
      %t1 = clg.load %arg1 : !cliff.ptr<f32> -> tensor<64x!cliff.multivector<1, unknown, f32, #space>, #layout>
      %ty = clg.load %y : !cliff.ptr<f32> -> tensor<64x!cliff.multivector<105, unknown, f32, #space>, #layout>

      %motor = cliff.rotate %t0, %t1 :
        tensor<64x!cliff.multivector<105, unknown, f32, #space>, #layout> *
        tensor<64x!cliff.multivector<1, unknown, f32, #space>, #layout> ->
        tensor<64x!cliff.multivector<113, unknown, f32, #space>, #layout>
      

      %2 = cliff.geo_prod %motor, %ty : tensor<64x!cliff.multivector<113, unknown, f32, #space>, #layout> * tensor<64x!cliff.multivector<105, unknown, f32, #space>, #layout> ->
                                        tensor<64x32x!cliff.multivector<255,  unknown, f32, #space>, #layout>

      %3 = cliff.reverse %motor : tensor<64x!cliff.multivector<113, unknown, f32, #space>, #layout> -> tensor<64x!cliff.multivector<113, unknown, f32, #space>, #layout>
      %out = cliff.geo_prod %2, %3 : tensor<64x32x!cliff.multivector<255,  unknown, f32, #space>, #layout> * tensor<64x!cliff.multivector<113, unknown, f32, #space>, #layout> ->
                                  tensor<64x32x!cliff.multivector<255,  unknown, f32, #space>, #layout>

      clg.store %pointer, %out : !cliff.ptr<f32>, tensor<64x32x!cliff.multivector<255,  unknown, f32, #space>, #layout>
      cliff.ret 
  }

}