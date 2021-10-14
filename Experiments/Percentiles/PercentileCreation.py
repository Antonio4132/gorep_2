import numpy as np
import pandas as pd

from scipy.stats import percentileofscore

def generate_percentiles(project_ID, subsampling, matrix, gene_names):
    """Generate the percentiles of a concrete subsampling of a project.

    Parameters
    ----------
    project_ID : str
        The ID of a project
    subsampling : dict
        The subsampling of the project of which we are going to create the percentiles
    matrix: scipy sparse matrix
        Sparse matrix of the entire project
    gene_names: pandas dataframe
        A dataframe with the names of the genes used in the matrix

    Returns
    -------

    results: pandas dataframe
        A dataframe with the percentile and the name of each gene
    results_info: pandas dataframe
        A dataframe of one row with the information of the sumsampling
    """
    sub_cells_index = subsampling['dataframe']['Assay'].index
    matrix = matrix.tocsr()
    sub_matrix = matrix[sub_cells_index].A

    sub_matrix_mean = np.mean(sub_matrix, axis=0)
    sub_matrix_mean_without_zeros = sub_matrix_mean[sub_matrix_mean != 0]

    sub_matrix_percentiles = [percentileofscore(sub_matrix_mean_without_zeros, x, 'weak') for x in sub_matrix_mean]

    gene_names['Gen_Name'] = gene_names['Gen_Name'].apply(lambda x: x.split('\t')[0])
    
    results = pd.DataFrame(data = {
        'gene_name': gene_names['Gen_Name'],
        'percentile': sub_matrix_percentiles
    })
    
    sub_copy = subsampling.copy()
    del sub_copy['dataframe']

    results_info = pd.DataFrame([{
        'project_id': project_ID,
        'metadata': str(sub_copy),
        'number_genes': len(sub_matrix_mean_without_zeros),
        'number_cells': len(subsampling['dataframe'])
    }])
    
    return results, results_info
