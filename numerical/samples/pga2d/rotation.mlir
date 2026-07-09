#space = #cliff.algebra<{p=2, q=0, r=1}>

cliff.func @rotation(%arg0 : tensor<NUM_ELSx!cliff.point<euclidean, true, #space>>, %arg1 : tensor<NUM_ELSx!cliff.scalar<#space>>, %pointer : !clg.ptr<f32>) {

    %0 = cliff.rotate %arg0, %arg1 :
        tensor<NUM_ELSx!cliff.point<euclidean, true, #space>> *
        tensor<NUM_ELSx!cliff.scalar<#space>> ->
        tensor<NUM_ELSx!cliff.motor<true, rotor, #space>>

        
    cliff.ret %0 : tensor<NUM_ELSx!cliff.motor<true, rotor, #space>>
}