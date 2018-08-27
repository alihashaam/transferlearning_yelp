import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from scipy.sparse import csr_matrix
import numpy as np
#############################################################################################################
def cm_normalize(cm):
    """
    Normalize Confusion Matrix

    Parameter:
        - cm: The confusion matrix to be normalized

    Return:
        - A normalized version of the confusion matrix
    """
    cm_normalized = []
    for row in cm:
        row_normalized = []
        for column in row:
            row_normalized.append( round(column * 1.0 / sum(row), 2) )
        cm_normalized.append(row_normalized)
    return cm_normalized
#############################################################################################################
def cm_plot(cm, labels, title, pdf_file, x_label='Predicted Label', y_label='True Label', normalize=True):
    """
    Plotting a confusion matrix and save the graph to pdf file

    Parameter:
        - cm: The confusion matrix to be plotted
        - labels: The labels of our classes
        - title: The title of the plot
        - pdf_file: Where to save the chart file
        - x_label: The label of x axes (Default='Predicted Label')
        - y_label: The label of y axes (Default='True Label')
        - normalize: Whether to normalize the matrix or not (Default=Ture)
    """
    df_cm = pd.DataFrame()
    if normalize:
        df_cm = pd.DataFrame(cm_normalize(cm), index=labels, columns=labels)
    else:
        df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    f = plt.figure(figsize = (10,7))
    ax = sn.heatmap(df_cm, annot=True)
    ax.xaxis.set_label_text(x_label)
    ax.yaxis.set_label_text(y_label)
    plt.title(title)
    f.savefig(pdf_file)
#############################################################################################################
def delete_from_csr(mat, row_indices=[], col_indices=[]):
    """
    Remove the rows (denoted by ``row_indices``) and columns (denoted by ``col_indices``) form the CSR sparse matrix ``mat``.
    WARNING: Indices of altered axes are reset in the returned matrix

    https://stackoverflow.com/questions/13077527/is-there-a-numpy-delete-equivalent-for-sparse-matrices/13078768
    """
    if not isinstance(mat, csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")

    rows = []
    cols = []
    if row_indices:
        rows = list(row_indices)
    if col_indices:
        cols = list(col_indices)

    if len(rows) > 0 and len(cols) > 0:
        row_mask = np.ones(mat.shape[0], dtype=bool)
        row_mask[rows] = False
        col_mask = np.ones(mat.shape[1], dtype=bool)
        col_mask[cols] = False
        return mat[row_mask][:,col_mask]
    elif len(rows) > 0:
        mask = np.ones(mat.shape[0], dtype=bool)
        mask[rows] = False
        return mat[mask]
    elif len(cols) > 0:
        mask = np.ones(mat.shape[1], dtype=bool)
        mask[cols] = False
        return mat[:,mask]
    else:
        return mat
#############################################################################################################
