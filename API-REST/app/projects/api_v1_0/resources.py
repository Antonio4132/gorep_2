import ast

from flask import request, Blueprint, jsonify
from flask_restful import Api

from ...db_conn import conn_controller as conn

projects_v1_0_bp = Blueprint('projects_v1_0_bp', __name__)

api = Api(projects_v1_0_bp)

@projects_v1_0_bp.route("/project", methods=['GET'])
def get_projects():
    """
    Get a list of projects using some parameters
    ---
    tags:
      - projects
    parameters:
      - in: query
        name: disease
        description: A disease on which a project has worked
        type: string
      - in: query
        name: cell_type
        description: Cell type studied in a project
        type: string
      - in: query
        name: sex
        description: Sex of the specimen studied on a project
        type: string

    responses:
      200:
        description: List of projects that meet selected parameters
    """

    params = {}

    disease = request.values.get('disease')
    if disease is not None:
        params['disease'] = disease

    cell_type = request.values.get('cell_type')
    if cell_type is not None:
        params['cell_type'] = cell_type

    sex = request.values.get('sex')
    if sex is not None:
        params['sex'] = sex

    projects = conn.get_projects(params)

    return jsonify(projects)


@projects_v1_0_bp.route("/project/info/<project_ID>", methods=['GET'])
def get_project_info(project_ID):
    """
    Get the information of a concrete project
    ---
    tags:
      - projects
    parameters:
      - in: path
        name: project_ID
        description: The ID of the project from which we want to get the information
        required: true
        type: string
    responses:
      200:
        description: Information of the project associated with the project_ID
    """

    project_info = conn.get_project_info(project_ID)

    return jsonify(project_info)


@projects_v1_0_bp.route("/project/metadata/<param>", methods=['GET'])
def get_project_metadata(param):
    """
        Get a list of possible values for a metadata parameter
        ---
        tags:
          - metadata
        parameters:
          - in: path
            name: param
            description: The metadata parameter of which you want to obtain the values
            required: true
            type: array
            items:
                type: string
                enum:
                    - disease
                    - cell_type
                    - organism_part
                    - analysis_protocol
                    - repository
                    - specie
                    - library
                    - biopsy_site
                    - project_ID
                default: disease
            collectionFormat: multi
        responses:
          200:
            description: List of values for metadata parameter
    """

    if param is None:
        return jsonify({'msg': 'param needed'})

    metadata_list = conn.get_project_metadata(param)

    return jsonify(metadata_list)


@projects_v1_0_bp.route("/project/downloads/<project_ID>", methods=['GET'])
def get_project_downloads(project_ID):
    """
        Get download links for a given project
        ---
        tags:
          - downloads
        parameters:
          - in: path
            name: project_ID
            description: Project ID of a specific project
            required: true
            type: string

        responses:
          200:
            description: List of download links of the project
    """

    if project_ID is None:
        return jsonify({'msg': 'project_ID needed'})

    downloads = conn.get_project_downloads(project_ID)

    return jsonify(downloads)


@projects_v1_0_bp.route("/percentiles", methods=['GET'])
def get_percentile():
    '''
        Get percentiles of the projects given a filter
        ---
        tags:
          - percentiles
        parameters:
          - in: query
            name: filters
            description: Criteria to filter percentiles from the search results. Each filter consists of a facet name and an array of facet values. Supported facet names are gen_names, cell_types and project_IDs.
            required: true
            schema:
                type: object
                default: {}
                properties:
                    gen_names:
                        type: array
                        items:
                            type:string
                        collectionFormat: csv
                    cell_types:
                        type: array
                        items:
                            type:string
                        collectionFormat: csv
                    specie:
                        type: array
                        items:
                            type:string
                        collectionFormat: csv
                    project_IDs:
                        type: array
                        items:
                            type:string
                        collectionFormat: csv
                example:
                    gen_names:
                        - ENSG00000287846
                        - ENSDARG00000034326
                    cell_types:
                        - MemoryBcell
                        - BloodCell
                    specie:
                        - HomoSapiens

        responses:
          200:
            description: List of download links of the project
    '''

    filters = request.values.get('filters')
    filters = ast.literal_eval(filters)

    gen_names = []
    cell_types = []
    project_IDs = []
    species = []

    print(filters)

    for key, value in filters.items():
        if key == 'gen_names':
            gen_names = value
        elif key == 'cell_types':
            cell_types = value
        elif key == 'project_IDs':
            project_IDs = value
        elif key == 'specie':
            species = value

    percentiles = conn.get_percentile(gen_names, cell_types, project_IDs, species)

    return jsonify(percentiles)
