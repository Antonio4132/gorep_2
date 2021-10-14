import requests
from . import fuseki_utils

# server_name = 'http://localhost:3030'  # URL in local
server_name = 'http://fuseki:3030' # URL in docker
service_name = 'ds'
request_url = server_name + '/' + service_name

def conn_alive():
    ans = requests.get(request_url, data={'query': '{}'})

    return ans.status_code == 200


def get_projects(params={}):
    where_content = "?project rdf:type a:Project . ?project a:PR.hasProjectID ?project_ID ."

    for key, value in params.items():
        if key == 'disease':
            where_content += " { ?project a:SPR.hasDisease ?disease . ?subClasses rdfs:subClassOf* a:" + value + ". ?disease rdf:type ?subClasses . } UNION { ?project a:SPR.hasDisease a:" + value + " . }"
        elif key == 'cell_type':
            where_content += " { ?project a:SPR.hasCellType ?cellType . ?subClasses rdfs:subClassOf* a:" + value + ". ?cellType rdf:type ?subClasses . } UNION { ?project a:SPR.hasCellType a:" + value + " . }"
        elif key == 'organism_part':
            where_content += " { ?project a:SPR.hasOrganismPart ?organismPart . ?subClasses rdfs:subClassOf* a:" + value + ". ?organismPart rdf:type ?subClasses . } UNION { ?project a:SPR.hasOrganismPart a:" + value + " . }"
        elif key == 'sex':
            where_content += " ?project a:SPR.hasSex \"" + value + "\" ."
        else:
            return {'msg': 'Param key not valid'}

    where_content += '''
        OPTIONAL { ?project a:SPR.hasProjectTitle ?projectTitle . }
        OPTIONAL { ?project a:PR.hasDescription ?description . }
        OPTIONAL { ?project a:SPR.hasAnalysisProtocol ?analysisProtocol . }
        OPTIONAL { ?project a:SPR.hasLibrary ?library . }
        OPTIONAL { ?project a:SPR.hasCellType ?cellType . }
        OPTIONAL { ?project a:SPR.hasSex ?sex . }
        OPTIONAL { ?project a:SPR.hasDisease ?disease . }
        OPTIONAL { ?project a:SPR.hasMinAge ?minAge . }
        OPTIONAL { ?project a:SPR.hasMinAge ?maxAge . }
        OPTIONAL { ?project a:SPR.hasAgeUnit ?ageUnit . }
        OPTIONAL { ?project a:SPR.hasOrganismPart ?organismPart . }
        OPTIONAL { ?project a:SPR.hasBiopsySite ?biopsySite . }
        OPTIONAL { ?project a:SPR.hasLaboratory ?laboratory . }
        OPTIONAL { ?project a:PR.hasInstitution ?institution . }
        OPTIONAL { ?project a:SPR.isPartOfRepository ?repository . }
        OPTIONAL { ?project a:SPR.isPartOfCollection ?collection . }
        OPTIONAL { ?project a:PR.hasDonorCount ?donorCount . }
        OPTIONAL { ?project a:PR.hasSpecimenCount ?specimenCount . }
    '''

    query = '''
        PREFIX a: <http://www.semanticweb.org/alicia/ontologies/2020/8/singleCellRepositories#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT DISTINCT
            ?project_ID
            ?projectTitle
            ?description
            ?analysisProtocol
            ?cellType
            ?sex
            ?minAge
            ?maxAge
            ?ageUnit
            ?disease
            ?organismPart
            ?biopsySite
            ?laboratory
            ?institution
            ?repository
            ?collection
            ?donorCount
            ?specimenCount
        WHERE { ''' + where_content + '}'

    #  print(query)

    response = requests.post(request_url, data={'query': query})

    projects = fuseki_utils.fuseki_to_dict(response)

    return projects


def get_project_metadata(metadata_param):
    metadata_param_with_URI = ['disease', 'cell_type', 'organism_part', 'biopsy_site']
    metadata_param_without_URI = ['sex', 'repository', 'library', 'specie', 'analysis_protocol', 'instrument']

    if metadata_param in metadata_param_with_URI:
        return get_project_metadata_with_URI(metadata_param)
    elif metadata_param in metadata_param_without_URI:
        return get_project_metadata_without_URI(metadata_param)

    return {'msg': 'Param key not valid'}


def get_project_metadata_with_URI(metadata_param):
    where_content = ""

    if metadata_param == 'disease':
        where_content += "{ ?term rdfs:subClassOf* a:Disease . } UNION { ?y rdfs:subClassOf* a:Disease . ?term rdf:type ?y } ."
    elif metadata_param == 'cell_type':
        where_content += "{ ?term rdfs:subClassOf* a:CellType . } UNION { ?y rdfs:subClassOf* a:CellType . ?term rdf:type ?y } ."
    elif metadata_param == 'organism_part' or metadata_param == "biopsy_site":
        where_content += "{ ?term rdfs:subClassOf* a:OrganismPart . } UNION { ?y rdfs:subClassOf* a:OrganismPart . ?term rdf:type ?y } ."

    query = '''
        PREFIX a: <http://www.semanticweb.org/alicia/ontologies/2020/8/singleCellRepositories#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT DISTINCT
            ?term
            ?URI
        WHERE { ''' + where_content + '\n OPTIONAL { ?term a:OR.hasURI ?URI } .\n }' + "\nORDER BY ?term"

    # print(query)

    response = requests.post(request_url, data={'query': query})

    metadata_list = fuseki_utils.fuseki_to_dict(response, dict_type="records")

    return metadata_list


def get_project_metadata_without_URI(metadata_param):
    where_content = ""

    if metadata_param == 'sex':
        where_content += " ?project a:SPR.hasSex ?x ."
    elif metadata_param == 'repository':
        where_content += " ?project a:SPR.isPartOfRepository ?x ."
    elif metadata_param == 'library':
        where_content += "{ ?x rdfs:subClassOf* a:Library . } UNION { ?y rdfs:subClassOf* a:Library . ?x rdf:type ?y }"
    elif metadata_param == 'specie':
        where_content += "{ ?x rdfs:subClassOf* a:Specie . } UNION { ?y rdfs:subClassOf* a:Specie . ?x rdf:type ?y }"
    elif metadata_param == 'analysis_protocol':
        where_content += "{ ?x rdfs:subClassOf* a:AnalysisProtocol . } UNION { ?y rdfs:subClassOf* a:AnalysisProtocol . ?x rdf:type ?y }"
    elif metadata_param == 'instrument':
        where_content += "{ ?x rdfs:subClassOf* a:InstrumentModel . } UNION { ?y rdfs:subClassOf* a:InstrumentModel . ?x rdf:type ?y }"
    elif metadata_param == 'project_ID':
        where_content += " ?project a:PR.hasProjectID ?x ."

    query = '''
                PREFIX a: <http://www.semanticweb.org/alicia/ontologies/2020/8/singleCellRepositories#>
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                SELECT DISTINCT
                    ?x
                WHERE { ''' + where_content + '}' + "ORDER BY ?x"

    # print(query)

    response = requests.post(request_url, data={'query': query})

    metadata_list = fuseki_utils.fuseki_to_list(response)

    return metadata_list


def get_project_downloads(project_ID):
    where_content = f"?project rdf:type a:Project . ?project a:PR.hasProjectID \"{project_ID}\" ."

    where_content += '''
            OPTIONAL { ?project a:PR.hasProjectID ?project_ID . }
            OPTIONAL { ?project a:SPR.hasClusteringLink ?clusteringLink . }
            OPTIONAL { ?project a:SPR.hasExperimentDesignLink ?experimentDesignLink . }
            OPTIONAL { ?project a:SPR.hasExperimentMetadataLink ?experimentMetadataLink . }
            OPTIONAL { ?project a:SPR.hasFilteredTPMLink ?filteredTPMLink . }
            OPTIONAL { ?project a:SPR.hasMarkerGenesLink ?markerGenesLink . }
            OPTIONAL { ?project a:SPR.hasMatrixLink ?matrixLink . }
            OPTIONAL { ?project a:SPR.hasNormalisedCountsLink ?normalisedCountsLink . }
            OPTIONAL { ?project a:SPR.hasRawCountsLink ?rawCountsLink . }
            OPTIONAL { ?project a:SPR.hasResultsLink ?resultsLink . }
        '''

    query = '''
            PREFIX a: <http://www.semanticweb.org/alicia/ontologies/2020/8/singleCellRepositories#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            SELECT
                ?project_ID
                ?clusteringLink
                ?experimentDesignLink
                ?experimentMetadataLink
                ?filteredTPMLink
                ?markerGenesLink
                ?matrixLink
                ?normalisedCountsLink
                ?rawCountsLink
                ?resultsLink
            WHERE { ''' + where_content + '}'

    # print(query)

    response = requests.post(request_url, data={'query': query})
    downloads = fuseki_utils.fuseki_to_dict(response)

    return downloads


def get_project_info(project_ID):
    query = '''
        PREFIX a: <http://www.semanticweb.org/alicia/ontologies/2020/8/singleCellRepositories#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT DISTINCT
          ?projectTitle
          ?specie
          ?disease
          ?cellType
          ?library
          ?instrument
          ?organismPart
          ?projectLink
          ?publicationLink
          ?publicationTitle
        WHERE { 
          ?project rdf:type a:Project . ''' + \
            f'?project a:PR.hasProjectID "{project_ID}" .' + \
            '''
          ?project a:SPR.hasProjectTitle ?projectTitle .
          ?project a:SPR.hasSpecie ?specie .
          ?project a:PR.hasProjectRepositoryLink ?projectLink .
          OPTIONAL { ?project a:SPR.hasOrganismPart ?organismPart . }
          OPTIONAL { ?project a:SPR.hasDisease ?disease . }
          OPTIONAL { ?project a:SPR.hasCellType ?cellType . }
          OPTIONAL { ?project a:SPR.hasLibrary ?library . }
          OPTIONAL { ?project a:SPR.hasInstrument ?instrument . }
          OPTIONAL { ?project a:PR.hasPublicationLink ?publicationLink . }
          OPTIONAL { ?project a:PR.hasPublicationTitle ?publicationTitle . }
        }
    '''

    print(query)
    response = requests.post(request_url, data={'query': query})
    print('recieved')

    project_info = fuseki_utils.fuseki_to_dict(response)

    return project_info
