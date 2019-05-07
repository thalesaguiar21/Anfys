import anfys.neural.builder as builder


_FEATURE_VECTOR_DIMENSION = 1


class ANFIS:

    def __init__(self, subset_size):
        self.subset_size = subset_size
        self.qtd_rules = 0
        self.qtd_inputs = 0
        self.fuzzysets = []
        self.cons_params = []
        self.prem_params = []
        self.linsys_coefs = []
        self.linsys_resul = []
        self.prem_mf = None
        self.regressor = None

    def fit_by_hybrid_learn(self, inputs, outputs, max_epochs):
        builder.configure_model(self, inputs.shape[_FEATURE_VECTOR_DIMENSION])
        epoch = 1
        while epoch <= max_epochs:
            for entry, output in zip(inputs, outputs):
                self._full_forwardpass_hybrid_learn(entry, output)
            epoch += 1

    def add_linsys_equation(self, coefs, result):
        self.linsys_coefs.append(coefs)
        self.linsys_resul.append(result)

    def l1sise(self):
        return self.qtd_mfs * self.qtd_mfs


class Sugeno(ANFIS):

    def __init__(self, subset_size):
        super().__init__(subset_size)
