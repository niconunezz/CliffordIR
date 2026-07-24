import numpy as np
from helpers import ObjectType, GeometricObj, GATestConfig

class MatrixGenerator():
    def __init__(self, GATestConfig):
        self.__conf_obj = GATestConfig
        self.__matrices = self.init_matrices()
        self.__matrix_c = self.init_matrix_c()

    def get_conf_obj(self):
        return self.__conf_obj

    def init_matrices(self):
        config = self.__conf_obj
        matrices = []
        for (mv, comp, objTy, geoTy) in zip(config.mvs,
                                            config.comps,
                                            config.objTys,
                                            config.geoTys):

            matrix = self.__initialize_random_matrix(comp)
            if objTy == ObjectType.Unknown:
                matrix = self.__initialize_random_matrix(comp)

            if (config.algebra == "pga2d"):
                print("ALGEBRA DETECTED")
                print("-"*50)
                print(f"objTy : {objTy}")
                print(f"geoTy : {geoTy}")
                if objTy is ObjectType.Euclidean:
                    print("EUCLIDEAN DETECTED")
                    match geoTy:
                        case GeometricObj.Point:
                            print("POINT DETECTED")
                            print("-"*50)
                            matrix[-1, :] = 1

            if (config.algebra == "pga3d"):
                if objTy == ObjectType.Euclidean:
                    match geoTy:
                        case GeometricObj.Line:
                            ...
                        

            matrices.append(matrix)

        return matrices

    def init_matrix_c(self):
        config = self.__conf_obj
        return np.empty((config.comps_c, config.N), dtype=np.float32)
    
    def __initialize_random_matrix(self, comps):
        config = self.__conf_obj
        return np.random.rand(comps, config.N).astype(np.float32)

    def __extract_comps(self):
        # get indices for a regular a*b (comps_c non zero components)
        config = self.__conf_obj

        i = 0
        params = []
        for (mv, matrix) in zip(config.mvs, self.__matrices):
            params.append(mv(*matrix[:, i]))

        mv_out = config.func(*params)
        vals = mv_out.value
        c_default_indices = np.nonzero(vals)[0]
        coeffs = vals[c_default_indices]
        assert (len(coeffs) == config.comps_c), "c_default_indices are wrong"

        return [np.nonzero(multivector.value)[0] for multivector in params], c_default_indices

    def generate_matrices(self):
        config = self.__conf_obj
        input_indices, c_default_indices = self.__extract_comps()
        for j in range(config.N):
            applied_mvs = [mv_func(*matrix[:, j])for (mv_func, matrix) in zip(config.mvs, self.__matrices)]

            mv_out= config.func(*applied_mvs)
            self.__matrix_c[:, j] = mv_out.value[c_default_indices]

            if (j == 0):
                for print_idx, mv in enumerate(applied_mvs):
                    print(f"mv{print_idx} : {mv}") 
                print(f"out : {mv_out}")

        for idx, matrix in enumerate(self.__matrices):
            np.savez(f"{config.saving_path}/matrix_{idx}", matrix)

        np.savez(f"{config.saving_path}/matrix_c", self.__matrix_c)
        return c_default_indices, input_indices