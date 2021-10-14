import pandas as pd
import numpy as np
import sys

from scipy.stats import percentileofscore
from scipy.io import mmread

def get_percentiles_from_project(project_ID):
    print("Reading files...")
    # Read project matrix and metadata
    matrix, metadata, cell_names = read_files(project_ID)
    delete_last_line()

    print("Getting cell type groups...")
    # Generate cell type groups with metadata
    cell_type_groups = get_cell_type_groups(metadata, cell_names)
    delete_last_line()

    print("Generating percentiles...")
    # Get mean percentiles of each cell type
    get_percentiles(matrix, cell_type_groups)
    delete_last_line()

    print("Percentiles created!")

    return cell_type_groups

def read_files(project_ID):
    #TODO Download matrix using ontology link
    download_matrix(project_ID)

    # Read matrix
    matrix_file_name = f'../SingleCell-Files/blood_downloads/{project_ID}.aggregated_filtered_normalised_counts.mtx'
    matrix = mmread(matrix_file_name).transpose()

    # Read metadata
    metadata = pd.read_csv(f'https://www.ebi.ac.uk/gxa/sc/experiment/{project_ID}/download?fileType=experiment-design&accessKey=', sep='\t')
    metadata = metadata[['Assay', 'Sample Characteristic[cell type]']]

    # Read cell names
    cells_file_name = f'../SingleCell-Files/blood_downloads/{project_ID}.aggregated_filtered_normalised_counts.mtx_cols'
    cell_names = pd.read_csv(cells_file_name, header=None, names=['Assay'])

    return matrix, metadata, cell_names

def download_matrix(project_ID):
    pass


def get_cell_type_groups(metadata, cell_names):
    # Merge both dataframes so we get the cells of the matrix with their corresponding type
    cell_types = pd.merge(
        cell_names,
        metadata,
        how="inner",
        on='Assay'
    )

    # Group by cell type
    grouped = cell_types.groupby(by='Sample Characteristic[cell type]')

    groups = []

    # For each group, assign its name and the index of the matrix for this type
    for name, group in grouped:
        groups.append({
            'name': name,
            'index': group.index
        })

    return groups


def get_percentiles(matrix, cell_type_groups):
    for group in cell_type_groups:

        print(group['name'])
        submatrix = matrix.A[group['index']]
        print(submatrix.shape)
        mean = np.mean(submatrix, axis=0)

        group['percentiles'] = [percentileofscore(mean, x, 'strict') for x in mean]

        delete_last_line()
        delete_last_line()


def delete_last_line():
    "Use this function to delete the last line in the STDOUT"

    #cursor up one line
    sys.stdout.write('\x1b[1A')

    #delete last line
    sys.stdout.write('\x1b[2K')
