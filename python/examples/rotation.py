from ..clifford.runtime.kernel import kernel
import torch
import numpy as np
from ..clifford.ga import multivector, CliffordAlgebra
from ..clifford import ga


@kernel
def rotate_kernel(x, angle, out_tensor):
    motor = ga.rotate(angle, x)
    return motor

def rotate():
    saving_path = "numerical/matrices/pga2d/rotation"
    x = np.load(f"{saving_path}/matrix_{0}.npz")['arr_0']
    angle = np.load(f"{saving_path}/matrix_{1}.npz")['arr_0']

    real_out = np.load(f"{saving_path}/matrix_c.npz")['arr_0']
    real_out = torch.tensor(real_out)
    out_tensor = torch.zeros((4, x.shape[-1]), dtype=torch.float32)
    x = torch.tensor(x)
    angle = torch.tensor(angle)

    x = multivector(104, x, CliffordAlgebra(2, 0, 1))
    angle = multivector(1, angle, CliffordAlgebra(2, 0, 1))
    out_tensor = multivector(105, out_tensor, CliffordAlgebra(2, 0, 1))

    rotate_kernel(x, angle, out_tensor)

    torch.cuda.synchronize()
    print(torch.allclose(out_tensor.data, real_out, rtol=1e-05, atol=1e-6))
if __name__ == "__main__":
    rotate()