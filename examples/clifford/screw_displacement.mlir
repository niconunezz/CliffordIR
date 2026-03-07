#space = #cliff.algebra<{p=3, q=0, r=1}>

cl.func @update_robot_arm_kernel(%points_ptr :!cl.ptr<f32>, %out_points_ptr :!cl.ptr<f32> , %angle : f32) -> {

    %a = arith.constant 0 : i32
    %b = arith.constant 1 : i32

    %p1 = cl.load(%points_ptr) : tensor<3xf32>

    %units = arith.constant 1 : i32

    %r = cl.rotor %angle, %a, %a, %b

    %t = cl.translator %units, %a, %b, %b
    %motor = cl.geo_prod %r, %t

    %tmp = cl.geo_prod %m, %p1
    %rev = cl.reverse %m
    %p2 = cl.geo_prod %rev

    cl.store %p2, $out_points_ptr : !cl.ptr<f32>
    cl.ret
}