

class Individual:

    def __init__(self, ID):
        self.ID = ID

        self.analysis_protocol = None
        self.biopsy_site = None
        self.cell_line_type = None
        self.disease = None
        self.instrument = None
        self.library = None
        self.model_organ = None
        self.organism_part = None
        self.preservation = None
        self.sample_type = None
        self.cell_type = None
        self.specie = None
        self.sample_status = None
        self.metastatic_site = None
        self.nucleic_acid = None

        self.biological_sex = None
        self.laboratory = None
        self.phenotype = None
        self.project_short_name = None
        self.project_title = None
        self.total_cell_counts = -1
        self.paired_end = None
        self.part_of_collection = None
        self.part_of_repository = None
        self.growth_condition = None

        self.clustering_link = None
        self.experiment_design_link = None
        self.experiment_metadata_link = None
        self.filtered_TPM_link = None
        self.marker_genes_link = None
        self.matrix_link = None
        self.normalised_counts_link = None
        self.raw_counts_link = None
        self.result_link = None

        super().__init__()

    def get_dict(self):
        individual_dict = {
            "ID": self.ID,
            "ObjectProperties": {
                "SPR.hasAnalysisProtocol": self.analysis_protocol,
                "SPR.hasBiopsySite": self.biopsy_site,
                "SPR.hasCellLineType": self.cell_line_type,
                "SPR.hasDisease": self.disease,
                "SPR.hasInstrument": self.instrument,
                "SPR.hasLibrary": self.library,
                "SPR.hasModel": self.model_organ,
                "SPR.hasOrganismPart": self.organism_part,
                "SPR.hasPreservation": self.preservation,
                "SPR.hasCellType": self.cell_type,
                "SPR.hasSpecie": self.specie,
                "SPR.hasSampleStatus": self.sample_status,
                "SPR.hasMetastaticSite": self.metastatic_site,
            },
            "DataProperties": {
                "SPR.hasSex": self.biological_sex,
                "SPR.hasPhenotype": self.phenotype,
                "SPR.hasTotalCellCount": self.total_cell_counts,
                "SPR.isPairedEnd": self.paired_end,
                "SPR.hasGrowthCondition": self.growth_condition,
                "SPR.hasSampleType": self.sample_type,
                "SPR.hasNucleicAcidSource": self.nucleic_acid,
            },
            "AnnotationProperties": {
                "SPR.hasLaboratory": self.laboratory,
                "SPR.hasProjectShortName": self.project_short_name,
                "SPR.hasProjectTitle": self.project_title,
                "SPR.isPartOfCollection": self.part_of_collection,
                "SPR.isPartOfRepository": self.part_of_repository,
                "SPR.hasClusteringLink": self.clustering_link,
                "SPR.hasExperimentDesignLink": self.experiment_design_link,
                "SPR.hasExperimentMetadataLink": self.experiment_metadata_link,
                "SPR.hasFilteredTPMLink": self.filtered_TPM_link,
                "SPR.hasMarkerGenesLink": self.marker_genes_link,
                "SPR.hasMatrixLink": self.matrix_link,
                "SPR.hasNormalisedCountsLink": self.normalised_counts_link,
                "SPR.hasRawCountsLink": self.raw_counts_link,
                "SPR.hasResultsLink": self.result_link
            }
        }

        return individual_dict
