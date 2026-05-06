import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def km(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) +1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # ind = linear_assignment(w.max() - w)
    ind = linear_sum_assignment(w.max() - w)

    result_dict = {key: value for key, value in zip(ind[0], ind[1])}
    y_pred_mapped = [result_dict[y] if y != -1 else -1 for y in y_pred]
    # print ("Accuracy: ", end = '')
    new_ind1 = ind[0].tolist()
    new_ind2 = ind[1].tolist()
    sum_acc = 0
    for k in range(len(new_ind1)):
        i = new_ind1[k]
        j = new_ind2[k]
        sum_acc += w[i, j]
    cluster_acc = sum_acc * 1.0 / y_pred.size
    return y_pred_mapped, cluster_acc


def evaluate_dbscan(y_true, y_pred):
    # 计算 Adjusted Rand Index (ARI)
    ari = adjusted_rand_score(y_true, y_pred)
    
    # 计算 Normalized Mutual Information (NMI)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    
    return ari, nmi