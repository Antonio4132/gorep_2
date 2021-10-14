from OntologyConversorAbstract import OntologyConversorAbstract
from Project import Project


class OntologyConversorGOREP(OntologyConversorAbstract):

    def init_map(self):
        return {}

    def format_concrete_specimen(self, raw_specimen, specimen_id):
        pass

    def format_concrete_project(self, raw_project, project_id):
        project = Project(project_id)

        # project_info_data
        project.project_id = raw_project['id']
        project.project_title = raw_project['title']
        project.project_short_name = raw_project['short_name']
        project.project_description = raw_project['description']
        project.update_date = raw_project['update_date']
        project.load_date = raw_project['load_date']
        project.array_express_id = raw_project['array_express_id']
        project.ena_id = raw_project['ENA_id']
        project.geo_series_id = raw_project['GEO_id']
        project.insdc_project_id = raw_project['INSD_project_id']
        project.insdc_study_id = raw_project['INSD_study_id']
        project.institutions = raw_project['institution']
        project.part_of_collection = raw_project['collection']
        project.part_of_repository = raw_project['repository']
        project.publication_title = raw_project['publication_title']
        project.publication_link = raw_project['publication_link']

        # project_metadata
        project.specie = raw_project['specie']
        project.cell_type = raw_project['cell_type']
        project.disease = raw_project['disease']
        project.organism_part = raw_project['organism_part']
        project.biopsy_site = raw_project['biopsy_site']
        project.metastatic_site = raw_project['metastatic_site']
        project.instrument = raw_project['instrument']
        project.library = raw_project['library']
        project.preservation = raw_project['preservation']
        project.biological_sex = raw_project['sex']
        # project = raw_project['min_age']
        # project = raw_project['max_age']
        # project = raw_project['age_unit']
        project.cell_line_type = raw_project['cell_line']
        project.nucleic_acid = raw_project['nucleic_acid_source']
        project.sample_type = raw_project['sample_type']

        self.project = project

    def parse_concrete(self, word):
        return word
