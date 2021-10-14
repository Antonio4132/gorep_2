from OntologyConversorAbstract import OntologyConversorAbstract
from Project import Project
from Specimen import Specimen


def norm2control(disease):
    if type(disease) is list:
        return list(map(norm2control, disease))

    if disease == 'Normal':
        disease = "Control"

    return disease


class OntologyConversorSCAE(OntologyConversorAbstract):

    def init_map(self):
        mapping_dict = {
            # Specie
            'PlasmodiumFalciparum3D7': 'PlasmodiumFalciparum',
            'BudTip': 'Bud',
            # Downloads
            'ExperimentDesignFile(TSVFormat)': 'ExperimentDesign',
            'ExperimentMetadata(SDRFAndIDFFilesArchive)': 'ExperimentMetadata',
            'ClusteringFile(TSVFormat)': 'Clustering',
            'FilteredTPMsFiles(MatrixMarketArchive)': 'FilteredTPMs',
            'MarkerGeneFiles(TSVFilesArchive)': 'MarkerGenes',
            'NormalisedCountsFiles(MatrixMarketArchive)': 'NormalisedCounts',
            'RawCountsFiles(MatrixMarketArchive)': 'RawCounts',
            # Sex
            'male-to-female transsexual': 'trans',
            'not available': 'notAvailable',
            'mixed sex population': 'mixed',
            'mixed (2 female, 1 male)': 'mixed',
            'not applicable': 'notApplicable',
            'mixed sex': 'mixed',
            # Organism Part
            'DorsalSkin': 'Skin',
            'ProstateGland': 'Prostate',
            'ZonaPellucida': 'PellucidZone',
            'brain without olfactory bulb': 'Brain',
            'primary visual area, layer…': 'PrimaryVisualArea',
            'CaudalGanglionicEminence': 'GanglionicEminence',
            'AnteriorNeuralTube': 'NeuralTube',
            'SplanchnicLayerOfLateralPlateMesoderm': 'Mesoderm',
            'SmoothMuscle,Peri-UrethralMesenchymeAndUrethralEpithelium': 'PeriUrethralMesenchyme',
            'VentralMedialGanglionicEminence': 'GanglionicEminence',
            'DorsalMedialGanglionicEminence': 'GanglionicEminence',
            'Embryonic/LarvalLymphGland': 'LymphNode',
            'PrimaryVisualArea,Layer1,Layer2/3AndLayer4': 'PrimaryVisualArea',
            'PrimaryVisualArea,Layer6A': 'PrimaryVisualArea',
            'PrimaryVisualCortex': 'PrimaryVisualArea',
            'PrimaryVisualArea,Layer5AndLayer6': 'PrimaryVisualArea',
            'PrimaryVisualArea,Layer1AndLayer2/3': 'PrimaryVisualArea',
            'PrimaryVisualArea,Layer2/3': 'PrimaryVisualArea',
            'PrimaryVisualArea,Layer4': 'PrimaryVisualArea',
            'PrimaryVisualArea,Layer6B': 'PrimaryVisualArea',
            'PrimaryVisualArea,Layer1': 'PrimaryVisualArea',
            'PrimaryVisualArea,Layer6': 'PrimaryVisualArea',
            'Layer1': 'RetinalNeuralLayer',
            'Layer2/3': 'RetinalNeuralLayer',
            'Layer4': 'RetinalNeuralLayer',
            'Cortical-LikeVentricle-Region1': 'CorticalLikeVentricle',
            'Cortical-LikeVentricle-Region2': 'CorticalLikeVentricle',
            'Cortical-LikeVentricle-Region3': 'CorticalLikeVentricle',
            'Cortical-LikeVentricle-Region4': 'CorticalLikeVentricle',
            'MetastasisToLymphNode': 'Metastasis',
            'BrainWithoutOlfactoryBulb': 'Brain',
            'NormalTissueAdjacentToTumour': 'NormalTissueAdjacentToTumor',
            'MixedCervicalAndThoracicVertebrae': ['CervicalVertebrae', 'ThoracicVertebrae'],
            'Fibroblasts': 'Fibroblast',
            'Embryonic/LarvalHemolymph': 'Hemolymph',
            'CardiacNon-MyocyteAndCardiomyocyte': ['CardiacNonMyocyte', 'Cardiomyocyte'],
            'AutopodRegion': 'Autopod',
            'Non-SmallCellLungCarcinoma': 'LungCarcinoma',
            'OlfactoryProjectionNeuronInnvervatingDC2Glomerulus': 'OlfatoryProjectionNeuron',
            'OlfactoryProjectionNeuronInnervatingVM2Glomerulus': 'OlfatoryProjectionNeuron',
            'ShortTermHematopoieticStemCell': 'HematopoieticStemCell',
            'OuterCortexOfKidney': 'CortexOfKidney',
            'RectalAdenocarcinoma': 'RectumAdenocarcinoma',
            'MainBronchus': 'Bronchus',
            'MouthFloor': 'OralCavity',
            'GingivaOfLowerJaw': 'Gingiva',
            'SupraglotticPartOfLarynx': 'Larynx',
            'PosteriorPartOfTongue': 'Tongue',
            'PrimaryVisualArea,Layer5': 'PrimaryVisualArea',
            'SkinOfBody': 'Skin',
            'DistalAirway': 'RespiratoryAirway',
            'RectosigmoidJunction': 'RectoSigmoidJunction',
            'OlfactoryBulb': 'OlfatoryBulb',
            'OlfactorySegmentOfNasalMucosa': 'OlfatoryBulb',
            # Biopsy Site
            'Ascites': 'AsciticFluid',
            'LeftSupraclavicularLymphNode': 'LymphNode',
            'AlveolusOfLung': 'Alveolus',
            'PrimaryTumor': 'Tumor',
            'PeripheralRegionOfRetina': 'Retina',
            'ProximalSmallIntestine': 'SmallIntestine',
            'MiddleAirway': 'RespiratoryAirway',
            'TumourCore': 'Tumor',
            'RightEye': 'Eye',
            'AdjacentToTumor': 'Tumor',
            'LaminaPropriaOfSmallIntestine': 'LaminaPropiaOfSmallIntestine',
            'SmoothMuscle': 'SmoothMuscleTissue',
            'PigmentedLayerOfRetina': 'PigmentedLayerOfRetinaAndOpticChoroid',
            'OpticChoroid': 'PigmentedLayerOfRetinaAndOpticChoroid',
            # Disease
            'ChronicPhaseChronicMyeloidLeukemia': 'MyeloidLeukemia',
            'BronchioalveolarCarcinoma;Non-SmallCellLungCancer': 'BronchioalveolarCarcinoma',
            'MetastaticBreastCancer': 'BreastCancer',
            'ProstateCarcinoma': 'ProstateCancer',
            'BreastCarcinoma': 'BreastCancer',
            'PancreaticNeoplasm': 'PancreaticCancer',
            'LungAdenocarcinoma': 'LungCarcinoma',
            'BrainGlioblastoma': 'Glioblastoma',
            'HypocellularMyelodysplasticSyndrome': 'MyelodysplasticSyndrome',
            'Crohn\'SDisease': 'CrohnsDisease',
            'HIVInfection': 'HIV',
            'OvarianSerousAdenocarcinoma': 'OvarianCarcinoma',
            'TypeIDiabetesMellitus': 'Type1DiabetesMellitus',
            'TypeIIDiabetesMellitus': 'Type2DiabetesMellitus',
            'Parkinson\'SDisease': 'ParkinsonsDisease',
            'ObstructiveSleepApneaSyndrome': 'ObstructiveSleepApnea',
            'BCellAcuteLymphoblasticLeukemia': 'AcuteLymphoblasticLeukemia',
            'HepatitisCInfection': 'HepatitisC',
            'SquamousCellLungCarcinoma': 'LungCarcinoma',
            'LargeCellLungCarcinoma': 'LungCarcinoma',
            'TumourEdge': 'Tumor',
            'TumourMiddle(InBetweenCoreAndEdgeSample)': 'Tumor',
            'TumourBorder': 'Tumor',
            'LargeTumor(>5Mm)': 'Tumor',
            'SmallTumor(<4Mm)': 'Tumor',
            'WetMacularDegeneration': 'MacularDegeneration',
            'Non-SmallCellLungCancer': 'NonSmallCellLungCancer',
            # Cell Type
            'OlfactoryProjectionNeuronInnervatingDA1,VA1DOrDC3Glomerulus': 'OlfatoryProjectionNeuron',
            'OlfactoryProjectionNeuron': 'OlfatoryProjectionNeuron',
            'NeutrophilAndMyeloidCell': ['Neutrophil', 'MyeloidCell'],
            'AdiposeTissueDerivedMesenchymalStemCell': 'MesenchymalStemCellOfAdipose',
            'MatureTCell': 'MatureTcell',
            'ThymicTCell': 'ThymicTcell',
            'HematopoieticStemCellAndThrombocyte': ['HematopoieticStemCell', 'Thrombocyte'],
            'EmbryonicNeuralBorderStemCell': 'EmbryonicStemCell',
            'Multi-LymphoidProgenitor': 'CommonLymphoidProgenitor',
            'PlantProtoplast': 'Protoplast',
            'CD8-PositiveT-Lymphocytes': 'CD8+AlphaBetaTcell',
            'CD8-Positive,Alpha-BetaTCell': 'CD8+AlphaBetaTcell',
            'MemoryBCell': 'MemoryBcell',
            'BCell': 'MemoryBcell',
            'MouseEmbryonicStemCell': 'EmbryonicStemCell',
            'HematopoieticStemCellAndThrombocyte': ['HematopoieticStemCell', 'Thrombocyte'],
            'Un-CryopreservedPeripheralBloodMononuclearCells(PBMCs)': 'PeripheralBloodMononuclearCell',
            'Lymphoid-PrimedMultipotentProgenitor': 'EarlyLymphoidProgenitor',
            'MixOfStromalFibroblastsAndEpithelialTumourCells': ['Fibroblast', 'EpithelialTumorCell'],
            'GranulocyteMacrophageProgenitor': 'GranulocyteMonocyteProgenitorCell',
            'Neuronal,GlialAndVascularCells': 'Neuron',
            'Marrow-DerivedBCell': 'MarrowDerivedBcell',
            'CD4-PositiveHelperTCell': 'CD4+HelperTcell',
            'HematopoieticStemCellAndHematopoieticMultipotentProgenitorCell': ['HematopoieticStemCell', 'HematopoieticMultipotentProgenitorCell'],
            'Pre-ConventionalDendriticCell': 'PreConventionalDendriticCell',
            'EpiblastCell': 'Epiblast',
            'MedialGanglionicEminence': 'GanglionicEminence',
            'NaiveThymus-DerivedCD4-Positive,Alpha-BetaTCell': 'NaiveThymusDerivedCD4+AlphaBetaTcell',
            'NeuralCrest-DerivedCell': 'NeuralCrestDerivedCell',
            'CardiacNon-Myocyte': 'CardiacNonMyocyte',
            'TransitionalStageBCell': 'TransitionalStageBcell',
            'NaiveBCell': 'NaiveBcell',
            'CD4-Positive,CD25-Positive,Alpha-BetaRegulatoryTCell': 'CD4+CD25+AlphaBetaRegulatoryTcell',
            'ExtraThymicAire-ExpressingCells': 'ExtraThymicAireExpressingCells',
            'GranulocyteMonocyteProgenitor': 'GranulocyteMonocyteProgenitorCell',
            'MarrowDerivedBCell': 'MarrowDerivedBcell',
            'InvasiveFront': 'Front',
            'Testis': 'Testes',
            'PosteriorIliacCrest': 'IliacCrest',
            'Megakaryocyte-ErythroidProgenitorCell,CommonMyeloidProgenitorAndGranulocyteMonocyteProgenitorCell': ['MegakaryocyteErythroidProgenitorCell', 'CommonMyeloidProgenitor'],
            'EffectorMemoryCD4-Positive,Alpha-BetaTCell': 'EffectorMemoryCD4+AlphaBetaTCell',
            'CD4-Positive,Alpha-BetaMemoryTCell': 'CD4+AlphaBetaMemoryTCell',
            'ClassicalMonocyte': 'Monocyte',
            'TCell': 'Tcell',
            'Glial': 'GlialCells',
            'Megakaryocyte-ErythroidProgenitorCell': 'MegakaryocyteErythroidProgenitorCell',
            'Neuronal': 'Neuron',
            # Preservation
            'FreshSpecimen': 'Fresh',
            # Library
            'Smart-Seq': 'Smart-seq',
            'Smart-Like': 'Smart-like',
            '10Xv2': '10Xv2Sequencing',
            '10XV2': '10Xv2Sequencing',
            'MixedPedalDigit3AndPedalDigit4': 'PedalDigit',
            'Digit4': 'PedalDigit',
            'Drop-Seq': 'Drop-seq',
            'Smart-Seq2': 'Smart-seq2',
            'Seq-Well': 'Seq-well',
            '10X5Prime': '10X5v2Sequencing',
            '10Xv3': '10xv3Sequencing',
            '10XV3': '10xv3Sequencing',
            # Stage
            '2-CellStageEmbryo': '2CellStageEmbryo',
            '4-CellStageEmbryo': '4CellStageEmbryo',
            '8-CellStageEmbryo': '8CellStageEmbryo',
            'MorulaCell': 'Morula',
        }
        return mapping_dict

    def format_concrete_specimen(self, raw_specimen, specimen_id):
        specimen = Specimen(specimen_id)
        specimen.part_of_repository = "SingleCellExpresionAtlas"
        specimen.project_title = raw_specimen['project_title']
        specimen.part_of_collection = raw_specimen['experiment_projects']
        specimen.total_cell_counts = raw_specimen['num_cells']
        specimen.sample_type = raw_specimen['sample_type']
        # specimen.assay = raw_specimen['cells']

        specimen = self.__specimen_info_to_specimen(specimen, raw_specimen['specimen_info'])

        self.specimen = specimen

    def format_concrete_project(self, raw_project, project_id):

        project = Project(project_id)

        project.part_of_repository = "SingleCellExpresionAtlas"

        project.project_id = raw_project['experimentAccession']
        project.project_title = raw_project['experimentDescription']
        project.specie = self.parse_word(raw_project['species'])
        project.load_date = raw_project['loadDate']
        project.update_date = raw_project['lastUpdate']
        project.total_cell_counts = raw_project['numberOfAssays']
        project.type = self.parse_word(raw_project['experimentType'])
        project.library = self.parse_word(raw_project['technologyType'])
        project.experimental_factor = self.parse_word(raw_project['experimentalFactors'])
        project.donor_count = raw_project['donors']
        project.sample_type = raw_project['sample_type']

        project.part_of_collection = self.parse_word(raw_project['experimentProjects'])
        project.repository_link = raw_project['repository_link']
        project.publication_title = raw_project['publication_title']
        project.publication_link = raw_project['publication_link']
        project.array_express_id = raw_project['ArrayExpress_ID']
        project.ena_id = raw_project['ENA_ID']

        project = self.__project_info_to_project(project, raw_project['project_info'])

        project.downloads_type = self.parse_word(self.__get_download_types(raw_project))

        project = self.__get_download_links(project.downloads_type, project)

        self.project = project

    def parse_concrete(self, word):
        aux = list(word.title())

        for i in range(len(word)):
            if word[i].isupper():
                aux[i] = word[i]

        aux = ''.join(aux).replace(' ', '')

        return aux

    def __get_download_types(self, raw_project):
        types = set()

        for download in raw_project['downloads']:
            for file in download['files']:
                if file['isDownload']:
                    description = file['description']
                    types.add(description)

        return list(types)


    def __get_download_links(self, download_types, project):
        if 'ExperimentDesign' in download_types:
            project.experiment_design_link = "https://www.ebi.ac.uk/gxa/sc/experiment/" + project.project_id + \
                                             "/download?fileType=experiment-design&accessKey="
        if 'ExperimentMetadata' in download_types:
            project.experiment_metadata_link = "https://www.ebi.ac.uk/gxa/sc/experiment/" + project.project_id + \
                                               "/download/zip?fileType=experiment-metadata&accessKey="
        if 'Clustering' in download_types:
            project.clustering_link = "https://www.ebi.ac.uk/gxa/sc/experiment/" + project.project_id +\
                                      "/download?fileType=cluster&accessKey="
        if 'FilteredTPMs' in download_types:
            project.filtered_TPM_link = "https://www.ebi.ac.uk/gxa/sc/experiment/" + project.project_id +\
                                        "/download/zip?fileType=quantification-filtered&accessKey="
        if 'MarkerGenes' in download_types:
            project.marker_genes_link = "https://www.ebi.ac.uk/gxa/sc/experiment/" + project.project_id +\
                                        "/download/zip?fileType=marker-genes&accessKey="
        if 'NormalisedCounts' in download_types:
            project.normalised_counts_link = "https://www.ebi.ac.uk/gxa/sc/experiment/" + project.project_id +\
                                             "/download/zip?fileType=normalised&accessKey="
        if 'RawCounts' in download_types:
            project.raw_counts_link = "https://www.ebi.ac.uk/gxa/sc/experiment/" + project.project_id +\
                                      "/download/zip?fileType=quantification-raw&accessKey="

        return project

    def __project_info_to_project(self, project, project_info):
        for key in project_info:
            if key == 'organism':
                project.specie = self.parse_word(project_info[key])
            elif key == 'sex':
                project.biological_sex = self.map_word(project_info[key])
            elif key == 'organism part':
                project.organism_part = self.parse_word(project_info[key])
            elif key == 'metastatic site':
                project.metastatic_site = self.parse_word(project_info[key])
            elif key == 'sampling site' or key == 'biopsy site':
                project.biopsy_site = self.parse_word(project_info[key])
            elif key == 'disease':
                disease = self.parse_word(project_info[key])
                disease = norm2control(disease)
                project.disease = disease
            elif key == 'cell type':
                project.cell_type = self.parse_word(project_info[key])
            elif key == 'biosource provider' or key == 'biomaterial_provi':
                project.laboratory = project_info[key]
            elif key == 'specimen with known storage state':
                project.preservation = self.parse_word(project_info[key])
            elif key == 'organismStatus':
                project.sample_status = project_info[key]
            elif key == 'growth condition':
                project.growth_condition = project_info[key]
            else:
                continue

        return project

    def __specimen_info_to_specimen(self, specimen, specimen_info):
        for key in specimen_info:
            if key == 'individual':
                specimen.specimen_ID = specimen_info[key]
            elif key == 'organism':
                specimen.specie = self.parse_word(specimen_info[key])
            elif key == 'age' or key == 'post conception age':
                specimen.min_age, specimen.max_age, specimen.age_unit = self.__process_age(specimen_info[key])
            elif key == 'sex':
                specimen.biological_sex = self.map_word(specimen_info[key])
            elif key == 'organism part':
                specimen.organism_part = self.parse_word(specimen_info[key])
            elif key == 'metastatic site':
                specimen.metastatic_site = self.parse_word(specimen_info[key])
            elif key == 'sampling site' or key == 'biopsy site':
                specimen.biopsy_site = self.parse_word(specimen_info[key])
            elif key == 'disease':
                disease = self.parse_word(specimen_info[key])
                disease = norm2control(disease)
                specimen.disease = disease
            elif key == 'cell type':
                specimen.cell_type = self.parse_word(specimen_info[key])
            elif key == 'biosource provider' or key == 'biomaterial_provi':
                specimen.laboratory = specimen_info[key]
            elif key == 'specimen with known storage state':
                specimen.preservation = self.parse_word(specimen_info[key])
            elif key == 'organismStatus':
                specimen.sample_status = specimen_info[key]
            elif key == 'growth condition':
                specimen.growth_condition = specimen_info[key]
            else:
                continue

        return specimen

    def __process_age(self, age):
        if type(age) is list:
            age_list = list(map(self.__process_age, age))
            age_list = zip(*age_list)
            age_list = list(map(list, age_list))
            
            if len(set(age_list[0])) > 1:
                age_list[0] = [x for x in age_list[0] if x != -1]

            return min(age_list[0]), max(age_list[1]), age_list[2]

        min_age = -1
        max_age = -1
        age_unit = 'year'

        for word in age.split(' '):

            try:
                # Parse numbers
                age_number = int(word)
                if min_age == -1:
                    min_age = age_number
                else:
                    max_age = age_number
            except ValueError:
                # Parse strings
                if word == 'gestational':
                    age_unit = 'gestationalWeek'
                if word == 'week' and age_unit != 'gestationalWeek':
                    age_unit = word
                if word == 'embryonic':
                    age_unit = 'embryonicDay'
                if word == 'day' and age_unit != 'embryonicDay':
                    age_unit = 'day'
                if word == 'hour' or word == 'month':
                    age_unit = word
                if word == 'applicable':
                    age_unit = 'notApplicable'
                if word == 'available':
                    age_unit = 'notAvailable'
        if max_age == -1:
            max_age = min_age

        return min_age, max_age, age_unit
