# author: hsl
# 2023/5/9 14:50
import numpy as np

class DatasetParameters:

    def __init__(self):
        self.dataset_list = []
        self.dataset_binkey = []

    def get_trace_set(self, trace_set_name):
        trace_list = self.get_trace_set_list()
        return trace_list[trace_set_name]

    def get_binkey_set(self, trace_set_name):
        binkey_list = self.get_binkey_set_list()
        return binkey_list[trace_set_name]

    def get_binkey_set_list(self):
        binkey_sm2 = r'0010010000100101000100101010101001001000010010101010010010000010100010101010' \
      r'00010100101000000001010010000101000100001010010100000000001010101001010101010' \
      r'10101010010101010001001010101000100010010010001001000010100010100100010010010' \
      r'01000010010101000010100101000010100101001010001001000001010000010000101001010' \
      r'0010100000101010010000100100101000010001001010100010001010000101'

        binkey_ecc = r'010101010000010101010000010101010000010101010000010101010000010101010000010101010000010101010000010101010000010101010000010101010000010101010000010101010000010101010000010101010000010101010000010101010000010101010000010101010000010101010000010101010000010101010000010101010000010101010000010101010000010101010000010101010000010101010000' \
                r'010101010000010101010000010101010000010101010000'

        self.dataset_binkey = {
            "sm2": binkey_sm2,
            "ecc_node": binkey_ecc,
            "ecc_randl": binkey_ecc,
            "ecc_fakeop": binkey_ecc
        }

        return self.dataset_binkey

    def get_trace_set_list(self):
        parameters_sm2 = {
            "n_clusters": 2,
            "eps": 1500,
            "min_samples": 10,
            "n_neighbors": 5,
            "branching_factor": 10,
            "xi": 0.1,
            "quantile": 0.3,
            "init_center": np.asarray([[5.95, 7.54], [-3.19, -3.30]], dtype=np.int),
            'type': "tied",
            'seeds':np.asarray([[4858, 2986], [-2665, -1253]], dtype=np.int)
        }

        parameters_ecc_nodefend = {
            "n_clusters": 2,
            "eps": 200,
            "min_samples": 5,
            "n_neighbors": 5,
            "max_eps": 10,
            "threshold": 1,
            "quantile": 0.4,
            "xi": 0.2,
            'type': "tied",
            "branching_factor": 10,
            'seeds': np.asarray([[-262, 2], [440, -4]], dtype=np.int),
            "init_center": np.asarray([[-262, 2], [440, -4]], dtype=np.int),
        }

        parameters_ecc_randdealy = {
            "n_clusters": 2,
            "eps": 100,
            "min_samples": 6,
            "n_neighbors": 2,
            "max_eps": 20,
            "threshold": 1,
            "xi": 0.3,
            "branching_factor": 10,
            'type': "tied",
            "quantile": 0.2,
            'seeds':np.asarray([[227, 172], [130, -196]], dtype=np.int),
            "init_center": np.asarray([[262, 351], [164, -229]], dtype=np.int)
        }

        parameters_ecc_fakeop = {
            "n_clusters": 2,
            "eps": 100,
            "min_samples": 5,
            "n_neighbors": 2,
            "max_eps": 20,
            "threshold": 1,
            "xi": 0.3,
            "branching_factor": 10,
            'type': "tied",
            "quantile": 0.4,
            'seeds': np.asarray([[227, 172], [130, -196]], dtype=np.int),
            "init_center": np.asarray([[262, 351], [164, -229]], dtype=np.int)
        }

        self.dataset_list = {
            "sm2": parameters_sm2,
            "ecc_node": parameters_ecc_nodefend,
            "ecc_randl": parameters_ecc_randdealy,
            "ecc_fakeop":parameters_ecc_fakeop
        }

        return self.dataset_list