import clifford as cl
import torch


@cl.compile
def update_robot_arm_kernel(points_ptr, out_points_ptr, angle):
    
    # for now imagine this loads an independent point from a 3d tensor.
    p1 = cl.load(points_ptr)

    # angle rotation about the z-axis
    r = cl.Rotor(angle, 0, 0, 1)
    # 1-unit translation in the yz direction.
    t = cl.Translator(units=1, x=0, y=1, z=1)
    # combine in a motor
    motor = r * t

    p2 = motor(p1)

    cl.store(p2, out_points_ptr)


def update_robot_arm():
    # B, x, y, z 
    update_robot_arm_kernel[](space = "PGA")
