from ..clifford.runtime.kernel import kernel
import torch
import numpy as np
from ..clifford.ga import multivector, CliffordAlgebra
from ..clifford import ga


@kernel
def rotate_kernel(x, angle, out_tensor):
    motor = ga.rotate(x, angle)
    return motor

def rotate():
    saving_path = "numerical/matrices/pga2d/rotation"
    x = np.load(f"{saving_path}/matrix_{0}.npz")['arr_0']
    angle = np.load(f"{saving_path}/matrix_{1}.npz")['arr_0']
    out_tensor = np.empty((4, x.shape[-1]), dtype=np.float32)    
    x = torch.tensor(x)
    angle = torch.tensor(angle)
    out_tensor = torch.tensor(out_tensor)

    x = multivector(104, x, CliffordAlgebra(2, 0, 1))
    angle = multivector(1, angle, CliffordAlgebra(2, 0, 1))
    out_tensor = multivector(105, out_tensor, CliffordAlgebra(2, 0, 1))


    print(f"x.shape : {x.shape}")
    print(f"angle.shape : {angle.shape}")

    rotate_kernel(x, angle, out_tensor)

if __name__ == "__main__":
    rotate()