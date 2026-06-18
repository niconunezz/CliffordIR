from generate_oracle import generate_geo_prods_matrices, generate_rotate_matrices, generate_rotate_appl_matrices
import numpy as np
import pycuda.driver as cuda
from clifford import Cl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

DEBUG = True
N = 4096

layout, blades = Cl(2, 0, 1, firstIdx=0)
e0 = blades['e0']
e1 = blades['e1']
e2 = blades['e2']
e01 = blades['e01']
e02 = blades['e02']
e12 = blades['e12']
e012 = blades['e012']

mv_point = lambda a, b, c, d: 0 + b*e02 + c*e01 + e12
mv_scalar = lambda a : a*e1*e1

CASES_ROTATE_APPL = [
    ("normal_case",       ),
]

def init_device(ptx, matrices, result, func_name):

    mod  = cuda.module_from_buffer(ptx.encode()+ b'\x00')
    
    # device init
    dev_matrices = [cuda.mem_alloc(m.nbytes) for m in matrices]
    result_dev = cuda.mem_alloc(result.nbytes)
    for dev_m, m in zip(dev_matrices, matrices) :  cuda.memcpy_htod(dev_m, m)

    func = mod.get_function(func_name)
    return func, dev_matrices, result_dev

def test_rotate_appl():
    cuda.init()
    ctx = cuda.Device(0).make_context()
    num_blocks = max(N//256, 1)
    num_threads = min(N, 1024)

    mv_x = mv_point
    mv_y = mv_point
    mv_alpha = mv_scalar
    comps_x = 4
    comps_y = 4
    comps_alpha = 1
    comps_c = 4

    X = np.empty((comps_x, N), dtype=np.float32)
    Y = np.empty((comps_y, N), dtype=np.float32)
    # point in (0, 1)
    X[0, :] = 0
    X[1, :] = 0
    X[2, :] = 1
    X[3, :] = 1

    # point in (1.2, 0.8)
    Y[0, :] = 0
    Y[1, :] = 1.2
    Y[2, :] = 0.8
    Y[3, :] = 1

    result_host = np.zeros(N * comps_c, dtype=np.float32)
    
    X = X.reshape(comps_x, N)[[0, 2, 1, 3]]
    Y = Y.reshape(comps_y, N)[[0, 2, 1, 3]]

    alpha = np.linspace(0, np.pi, N).reshape(1, N).astype(np.float32)

    # generate ptx for case
    # subprocess.run(["./numerical/compile_case.sh", str(0), str(0), str(0), "output.ptx"])

    with open("output.ptx", "r") as f:
        ptx = f.read()
    
    func, dev_matrices, result_dev = init_device(ptx, [X, Y, alpha], result_host, "rotation")
    X_dev, Y_dev, alpha_dev = dev_matrices 

    func(X_dev, Y_dev, alpha_dev, result_dev,
     block=(num_threads, 1, 1),
     grid=(num_blocks, 1, 1))
    
    cuda.memcpy_dtoh(result_host, result_dev)

    ctx.pop()

    return result_host


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

if __name__ == '__main__':

    xx = (1, 0)
    rotated_points = test_rotate_appl().reshape(4, N)

    out_pts = []
    for i in range(N):
        pt = mv_point(0, rotated_points[1, i], rotated_points[2, i], 1).value
        w = pt[-2]
        x = pt[-3] / w
        y = pt[-4] / w
        out_pts.append((x, y))

    fig, ax = plt.subplots(figsize=(6, 6), dpi=80)

    xs = [p[0] for p in out_pts]
    ys = [p[1] for p in out_pts]

    ax.set_xlim(-1.5, 3)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')

    fixed_point = ax.scatter(*xx)
    fixed_point_z = ax.scatter(0.8, 1.2)


    moving_point = ax.scatter([], [])
    line, = ax.plot([], [])

    def update(frame):
        p = out_pts[frame]

        moving_point.set_offsets([[p[0], p[1]]])

        line.set_data(
            [xx[0], p[0]],
            [xx[1], p[1]]
        )

        return moving_point, line

    ani = FuncAnimation(
        fig,
        update,
        frames=len(out_pts),
        interval=5,
        blit=True
    )
    ani.save("rotation.gif", writer="ffmpeg", fps=60)