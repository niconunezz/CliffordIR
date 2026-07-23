import numpy as np
from helpers import ObjectType, GeometricObj

class MatrixGenerator():
    def __init__(self, mvs, comps, objTys, geoTys,
                 comps_c, objTy_c, geoTy_c, algebra : str, N : int, func,
                 saving_path = "numerical/matrices"):
        self.__algebra = algebra
        self.__num_els = N
        self.__mvs = mvs
        self.__comps = comps
        self.__objTys = objTys
        self.__geoTys = geoTys
        self.__matrices = self.init_matrices()
        self.__comps_c = comps_c
        self.__objTy_c = objTy_c
        self.__geoTy_c = geoTy_c
        self.__func = func
        self.__saving_path = saving_path 
        self.__matrix_c = self.init_matrix_c()

    def get_mvs(self):
        return self.__mvs
    def get_comps(self):
        return self.__comps

    def get_comp_c(self):
        return self.__comps_c
    
    def get_num_els(self):
        return self.__num_els

    def get_func(self):
        return self.__func

    def get_geoTys(self):
        return self.__geoTys 

    def get_saving_path(self):
        return self.__saving_path

    def init_matrices(self):
        matrices = []
        for (mv, comp, objTy, geoTy) in zip(self.__mvs,
                                            self.__comps,
                                            self.__objTys,
                                            self.__geoTys):

            matrix = self.__initialize_random_matrix(comp)
            if objTy == ObjectType.Unknown:
                matrix = self.__initialize_random_matrix(comp)

            if (self.__algebra == "pga2d"):
                if objTy == ObjectType.Euclidean:
                    match geoTy:
                        case GeometricObj.Point:
                            matrix[-1, :] = 1

            if (self.__algebra == "pga3d"):
                if objTy == ObjectType.Euclidean:
                    match geoTy:
                        case GeometricObj.Line:
                            ...
                        

            matrices.append(matrix)

        return matrices

    def init_matrix_c(self):
        # if self.__objTy_c == ObjectType.Unknown:
        return np.empty((self.__comps_c, self.__num_els), dtype=np.float32)
    
    def __initialize_random_matrix(self, comps):
        return np.random.rand(comps, self.__num_els).astype(np.float32)

    def __extract_comps(self):
        # get indices for a regular a*b (comps_c non zero components)
        i = 0

        params = []
        for (mv, matrix) in zip(self.__mvs, self.__matrices):
            params.append(mv(*matrix[:, i]))

        mv_out = self.__func(*params)
        vals = mv_out.value
        c_default_indices = np.nonzero(vals)[0]
        coeffs = vals[c_default_indices]
        assert (len(coeffs) == self.__comps_c), "c_default_indices are wrong"

        return [np.nonzero(multivector.value)[0] for multivector in params], c_default_indices

    def generate_matrices(self):

        input_indices, c_default_indices = self.__extract_comps()
        for j in range(self.__num_els):
            applied_mvs = [mv_func(*matrix[:, j])for (mv_func, matrix) in zip(self.__mvs, self.__matrices)]

            mv_out= self.__func(*applied_mvs)
            self.__matrix_c[:, j] = mv_out.value[c_default_indices]

            if (j == 0):
                for print_idx, mv in enumerate(applied_mvs):
                    print(f"mv{print_idx} : {mv}") 
                print(f"out : {mv_out}")

        for idx, matrix in enumerate(self.__matrices):
            np.savez(f"{self.__saving_path}/matrix_{idx}", matrix)

        np.savez(f"{self.__saving_path}/matrix_c", self.__matrix_c)
        return c_default_indices, input_indices