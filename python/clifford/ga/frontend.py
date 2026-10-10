from .._C.libclifford import ir

class CliffordFrontend:
    
    def __init__(self, builder):
        self.builder = builder


    def rotate(self, angle, x):
        return self.builder.create_rotate(angle, x)

    def geo_prod(self, x, y):
        return self.builder.create_geo_prod(x, y)

    def reverse(self, x):
        return self.builder.create_reverse(x)
    