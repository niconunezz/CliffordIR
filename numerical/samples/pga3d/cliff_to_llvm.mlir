#space = #cliff.algebra<{p=3, q=0, r=1}>


cliff.func @rotation(%arg0 : tensor<NUM_ELSx!cliff.line<euclidean, true, #space>>,
                    %arg1 : tensor<NUM_ELSx!cliff.point<euclidean, true, #space>>,
                    %time : tensor<NUM_ELSx!cliff.scalar<#space>>,
                    %store : tensor<NUM_ELSx!cliff.point<euclidean, true, #space>>) -> tensor<NUM_ELSx!cliff.point<euclidean, true, #space>> {

    
    %tmp = cliff.geo_prod %time, %arg0 : 
        tensor<NUM_ELSx!cliff.scalar<#space>> * 
        tensor<NUM_ELSx!cliff.line<euclidean, true, #space>> -> tensor<NUM_ELSx!cliff.line<euclidean, true, #space>>

    %motor = cliff.exp %tmp : tensor<NUM_ELSx!cliff.line<euclidean, true, #space>> -> tensor<NUM_ELSx!cliff.motor<true, rotor, #space>>

    %out = cliff.sandwich %motor, %arg1 : 
        tensor<NUM_ELSx!cliff.motor<true, rotor, #space>> *
        tensor<NUM_ELSx!cliff.point<euclidean, true, #space>> -> tensor<NUM_ELSx!cliff.point<euclidean, true, #space>>

    cliff.ret %out : tensor<NUM_ELSx!cliff.point<euclidean, true, #space>>
}
