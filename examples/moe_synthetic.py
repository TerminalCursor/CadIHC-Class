from histclass.tools.moe.synthetic import *
from histclass.tools.moe.sampling import *
from numpy import array, zeros, diag
from numpy.random import dirichlet, rand

N_CLASSES = 3
N_SAMPLES = 1000
N_MODELS = 5

model_confidences = dirichlet(
    alpha=array(range(N_CLASSES)) * 0 + 1,
    size=N_MODELS,
)
model_confidences = rand(N_MODELS, N_CLASSES)
model_confidences[0, :] = array([0.25, 0.5, 0.25])
model_confidences[0, :] = array([0.75, 0.75, 0.75])
model_confidences = array(
    [
        [0.9, 0.9, 0.7],
        [0.7, 0.9, 0.9],
        # [0.1, 0.1, 0.1],
        [0.3, 0.3, 0.3],
        [0.5, 0.01, 0.9],
    ]
)
N_MODELS = model_confidences.shape[0]
print(model_confidences)

_smm = SyntheticMoEModels(model_confidences, n_classes=N_CLASSES)
_smd = SyntheticMoEData.generate(_smm, n_samples=N_SAMPLES)

# Verifying the label-based performance per model
_smm.get_accuracy(_smd)
print(_smm.model_confidences)
print(_smm.model_accuracies)

for f in [smoe_mix_1, smoe_gate_1, smoe_bayes_1, smoe_bayes_2]:
    label_accuracies = zeros((_smm.n_classes, _smm.n_classes))
    for _ in range(1000):
        sample = smoe_run(_smm, _smd, f)
        if sample is None:
            continue
        p_i, l_i = sample
        label_accuracies[l_i, p_i] += 1
    print(f)
    print(label_accuracies)
    print(diag(label_accuracies) / label_accuracies.sum(-1))
    print(diag(label_accuracies).sum() / label_accuracies.sum())
