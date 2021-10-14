import os

import whoosh.index as index
import pandas as pd
import requests

from whoosh.qparser import QueryParser
from ..common import utils

from . import conn_controller as conn


def get_percentile(gen_names=[], cell_types=[], project_IDs=[]):
    ix = index.open_dir("../SingleCell-Files/index")
    qp = QueryParser("content", ix.schema)
    genes_project_IDs = []
    cell_types_project_IDs = []

    # Search for projects with that meet the filter conditions (gen_names and cell_types)
    for gen_name in gen_names:

        q = qp.parse(gen_name)

        with ix.searcher() as s:
            results = s.search(q, limit=None)
            for result in results:
                genes_project_IDs.append(result['title'])

    for cell_type in cell_types:

        q = qp.parse(cell_type)

        with ix.searcher() as s:
            results = s.search(q, limit=None)
            for result in results:
                cell_types_project_IDs.append(result['title'])

    percentile_project_IDs = utils.intersection(genes_project_IDs, cell_types_project_IDs)

    percentile_projects = []
    key_items = ['project_ID', 'gen_name'] + cell_types

    if not percentile_project_IDs:
        percentile_project_IDs = project_IDs

    print(percentile_project_IDs)

    # For each project parse the response
    for percentile_project_ID in percentile_project_IDs:
        # Filter with project ID
        if project_IDs and percentile_project_ID not in project_IDs:
            continue

        # If project is not in the index
        filename = f'../SingleCell-Files/percentiles/{percentile_project_ID}.percentiles.csv'
        if not os.path.isfile(filename):
            continue

        # Read percentile file
        df = pd.read_csv(filename)

        # Filter gene names
        if gen_names:
            df = df[df['gen_name'].isin(gen_names)]
        print("project_info")

        project_info = conn.get_project_info(percentile_project_ID)
        print(project_info)

        # Loop over gen-project
        for df_dict in df.to_dict('records'):
            df_dict.update({'project_ID': percentile_project_ID})

            if cell_types:
                df_dict = {k: v for k, v in df_dict.items() if k in key_items}

            df_dict.update(project_info)
            percentile_projects.append(df_dict)

    return percentile_projects
