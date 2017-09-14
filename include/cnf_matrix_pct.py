
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def compute_cnf_matrix(true_label,prediction):
    cnf_matrix = confusion_matrix(true_label, prediction)
    cnf_matrix_pct = []
    for row in cnf_matrix:
        cnf_matrix_pct.append([x * 100. / sum(row) for x in row])
    cnf_matrix_pct = np.array(cnf_matrix_pct)
    return cnf_matrix,cnf_matrix_pct

