from . import fuseki_con as fuseki
from . import percentiles_local as percentiles
from . import Postgres_Controller

psql = Postgres_Controller.PostgresController()

# region Fuseki functions
def get_projects(params={}):
    return fuseki.get_projects(params)


def get_project_info(project_ID):
    return fuseki.get_project_info(project_ID)


def get_project_metadata(metadata_param):
    metadata_param_with_URI = ['disease', 'cell_type', 'organism_part', 'biopsy_site']
    metadata_param_without_URI = ['sex', 'repository', 'library', 'specie', 'analysis_protocol', 'instrument', 'project_ID']

    if metadata_param in metadata_param_with_URI:
        return fuseki.get_project_metadata_with_URI(metadata_param)
    elif metadata_param in metadata_param_without_URI:
        return fuseki.get_project_metadata_without_URI(metadata_param)

    return {'msg': 'Param key not valid'}


def get_project_downloads(project_ID):
    return fuseki.get_project_downloads(project_ID)

# endregion


def get_percentile(gen_names=[], cell_types=[], project_IDs=[], species=[]):
    return psql.get_percentile(gen_names, cell_types, project_IDs, species)
