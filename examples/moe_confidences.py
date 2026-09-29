import pandas as pd
from numpy import unique

raw_data = pd.read_pickle("MoeData.pickle")

# Finding confidences:
# Given a model (name, category, id)
#  Get all of the training data: usage = 1
#  For each label (l):
#    Get the training data with that label
#      Compute true positives  (label == pred)
#      Compute false negatives (label != pred)
#      TPR = TP / (TP + FN)
#      RawConfidence = TPR
raw_data["FullName"] = (
    raw_data["ModelName"] + "2" + raw_data["Category"] + "." + raw_data["ModelId"]
)

moe_confidences = []
model_names = set(unique(raw_data["FullName"]))
for name in sorted(model_names):
    model_data = raw_data[raw_data["FullName"] == name]
    model_data = model_data[model_data["Usage"] == 1]
    categories = unique(model_data["Category"])
    assert len(categories) == 1
    n_classes = 2 if categories[0] == "tn" else 3
    labels = model_data["Label"]
    preds = model_data["Prediction"]
    confidences = []
    for label in range(n_classes):
        lmask = labels == label
        llabel = labels[lmask]
        plabel = preds[lmask]
        COUNT = lmask.shape[0]
        TP = (llabel == plabel).sum()
        FN = (llabel != plabel).sum()
        TPR = float(TP / (TP + FN))
        confidences.append(TPR)
    if len(confidences) == 2:
        confidences.append(-1.0)
    moe_confidences.append((name, categories[0], *confidences))
moe_confidences = pd.DataFrame(
    moe_confidences, columns=["Name", "Category", "Conf-0", "Conf-1", "Conf-2"]
)
print(moe_confidences)
moe_confidences.to_pickle("MoeConfidences.csv")
moe_confidences.to_pickle("MoeConfidences.pickle")
