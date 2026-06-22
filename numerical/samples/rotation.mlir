#layout = #clg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [], block = []}>

#space = #cliff.algebra<{p=2, q=0, r=1}>


cliff.func @rotation(%arg0 : !clg.ptr<f32>, %arg1 : !clg.ptr<f32>, %pointer : !clg.ptr<f32>) {
    
    %t0 = clg.load %arg0 : !clg.ptr<f32> -> tensor<64x!cliff.multivector<105, unknown, f32, #space>, #layout>
    %t1 = clg.load %arg1 : !clg.ptr<f32> -> tensor<64x!cliff.multivector<1, unknown, f32, #space>, #layout>

    %0 = cliff.rotate %t0, %t1 :
        tensor<64x!cliff.multivector<105, unknown, f32, #space>, #layout> *
        tensor<64x!cliff.multivector<1, unknown, f32, #space>, #layout> ->
        tensor<64x!cliff.multivector<105, unknown, f32, #space>, #layout>
        
    clg.store %pointer, %0 : !clg.ptr<f32>, tensor<64x!cliff.multivector<105, unknown, f32, #space>, #layout>
    cliff.ret
}