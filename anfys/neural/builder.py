from anfys.fuzzy.subsets import FuzzySet


def configure(anfis, qtd_inputs):
    pass


def _build_subsets(anfis, qtd_inputs):
    anfis.fuzzysets = [FuzzySet(anfis.prem_mf) for _ in range(qtd_inputs)]
