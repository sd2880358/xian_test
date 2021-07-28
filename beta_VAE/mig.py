import numpy as np
import sklearn
from libsvm.svmutil import *



def compute_mig(model, data, labels):
    score_dict = {}
    mus, _ = model.encode(data)
    m = discrete_mutual_info(mus, labels)
    assert m.shape[0] == mus.shape[0]
    assert m.shape[1] == labels.shape[0]
    entropy = discrete_entropy(labels)
    sorted_m = np.sort(m, axis=0)[::-1]
    score_dict["discrete_mig"] = np.mean(
        np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:]))
    return score_dict

def discrete_mutual_info(mus, ys):
  """Compute discrete mutual information."""
  num_codes = mus.shape[0]
  num_factors = ys.shape[0]
  m = np.zeros([num_codes, num_factors])
  for i in range(num_codes):
    for j in range(num_factors):
      m[i, j] = sklearn.metrics.mutual_info_score(ys[j, :], mus[i, :])
  return m

def discrete_entropy(ys):
  """Compute discrete mutual information."""
  num_factors = ys.shape[0]
  h = np.zeros(num_factors)
  for j in range(num_factors):
    h[j] = sklearn.metrics.mutual_info_score(ys[j, :], ys[j, :])
  return h