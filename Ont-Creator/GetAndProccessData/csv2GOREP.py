import pandas as pd


def csv2GOREP(project_info, metadata_table):
    project_dict = {}
    project_info_no_na = project_info.fillna('null')

    dict_copy = project_info_no_na.to_dict('records')

    for row in dict_copy:
        key = row['key']
        values = row['value'].split(';')

        project_dict[key] = [value for value in values if value != 'null']

    metadata_table_no_na = metadata_table.fillna('null')

    for key in metadata_table_no_na:
        if key == 'assay':
            continue

        values = metadata_table_no_na[key].unique()
        project_dict[key] = [value for value in values if value != 'null']

    return project_dict
