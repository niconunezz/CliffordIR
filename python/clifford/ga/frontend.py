from .._C.libclifford import ir

class CliffordFrontend:
    
    def __init__(self, builder):
        self.builder = builder


    def rotate(self, angle, x):
        return self.builder.create_rotate(angle, x)
    