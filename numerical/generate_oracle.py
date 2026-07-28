import numpy as np
from helpers import ObjectType, GeometricObj

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

            elif (config.algebra == "pga2d"):
                if objTy is ObjectType.Euclidean:
                    match geoTy:
                        case GeometricObj.Point:
                            matrix[-1, :] = 1

            elif (config.algebra == "pga3d"):
                if objTy == ObjectType.Euclidean:
                    match geoTy:
                        case GeometricObj.Line:
                            matrix = self.__random_lines(mv, config.N)

            matrices.append(matrix)

        return matrices

    def init_matrix_c(self):
        config = self.__conf_obj
        return np.empty((config.comps_c, config.N), dtype=np.float32)
    
    def __initialize_random_matrix(self, comps):
        config = self.__conf_obj
        return np.random.rand(comps, config.N).astype(np.float32)

    def __random_lines(self, line_func, N: int) -> np.ndarray:
        # Coeficientes libres
        a = np.random.randn(N)
        b = np.random.randn(N)
        d = np.random.randn(N)
        e = np.random.randn(N)
        f = np.random.randn(N)

        # Evitar divisiones por cero
        while np.any(np.abs(d) < 1e-12):
            mask = np.abs(d) < 1e-12
            d[mask] = np.random.randn(mask.sum())

        # Impone la condición de Plücker: af - be + cd = 0
        c = (b * e - a * f) / d

        comps = np.vstack([a, b, c, d, e, f])  # (6, N)

        # Normalizar cada columna para que L² = -1
        for i in range(N):
            L = line_func(*comps[:, i])
            comps[:, i] /= np.sqrt(abs((L * L).value[0]))
        return comps.astype(np.float32)


    def __extract_comps(self):
        # get indices for a regular a*b (comps_c non zero components)
        config = self.__conf_obj

        i = 0
        params = []
        for (mv, matrix) in zip(config.mvs, self.__matrices):
            params.append(mv(*matrix[:, i]))

        mv_out = config.func(*params)
        vals = mv_out.value
        #todo: find a way decimals can be higher, at least ~10
        c_default_indices = np.nonzero(np.round(vals, decimals=7))[0]
        assert (len(c_default_indices) == config.comps_c), f"len(coeffs) = {len(c_default_indices)} doesnt match comps_c = {config.comps_c}"

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