import clifford as cl
import torch


@cl.compile
def update_robot_arm_kernel(points_ptr, axis_start, axis_end, angle, distance):
    axis = cl.Line(axis_start, axis_end)
    motor = cl.Screw(line=axis, angle=angle, pitch=distance)

    p = cl.load(points_ptr)

    out = motor >> p
    cl.store(points_ptr, out)


def update_robot_arm():
    # B, x, y, z 
    points = ...
    grid = ()
    update_robot_arm_kernel[grid](points)