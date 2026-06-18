#space = #cliff.algebra<{p=2, q=0, r=1}>


cliff.func @rotation(%arg0 : tensor<NUM_ELSx!cliff.point<euclidean, true, #space>>,
                    %arg1 : tensor<NUM_ELSx!cliff.point<euclidean, true, #space>>,
                    %time : tensor<NUM_ELSx!cliff.scalar<#space>>,
                    %store : tensor<NUM_ELSx!cliff.point<euclidean, true, #space>>) -> tensor<NUM_ELSx!cliff.point<euclidean, true, #space>> {

    
    %tmp = cliff.geo_prod %time, %arg0 : 
        tensor<NUM_ELSx!cliff.scalar<#space>> * 
        tensor<NUM_ELSx!cliff.point<euclidean, true, #space>> -> tensor<NUM_ELSx!cliff.point<euclidean, true, #space>>

    %motor = cliff.exp %tmp : tensor<NUM_ELSx!cliff.point<euclidean, true, #space>> -> tensor<NUM_ELSx!cliff.motor<true, #space>>

    %out = cliff.sandwich %motor, %arg1 : 
        tensor<NUM_ELSx!cliff.motor<true, #space>> *
        tensor<NUM_ELSx!cliff.point<euclidean, true, #space>> -> tensor<NUM_ELSx!cliff.point<euclidean, true, #space>>

    cliff.ret %out : tensor<NUM_ELSx!cliff.point<euclidean, true, #space>>
}