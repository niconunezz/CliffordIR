from ..clifford.runtime.kernel import kernel
import torch
import numpy as np
from ..clifford.ga import multivector, CliffordAlgebra



@kernel
def rotate_kernel(x, angle):
    motor = ga.rotate(x, angle)
    return motor

def rotate():
    saving_path = "numerical/matrices/pga2d/rotation"
    x = np.load(f"{saving_path}/matrix_{0}.npz")['arr_0']
    angle = np.load(f"{saving_path}/matrix_{1}.npz")['arr_0']

    x = torch.tensor(x)
    angle = torch.tensor(angle)
    x = multivector(b'0011000', x, CliffordAlgebra(2, 0, 1))
    angle = multivector(b'1', angle, CliffordAlgebra(2, 0, 1))

    print(f"x.shape : {x.shape}")
    print(f"angle.shape : {angle.shape}")

    rotate_kernel(x, angle)

if __name__ == "__main__":
    rotate()