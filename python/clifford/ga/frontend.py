from .._C.libclifford import ir

class CliffordFrontend:
    
    def __init__(self, builder):
        self.builder = builder


    def rotate(self, x, angle):
        return self.builder.create_rotate(x, angle)
    