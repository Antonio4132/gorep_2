from Individual import Individual


class Specimen(Individual):

    def __init__(self, specimen_id):
        self.specimen_ID = None
        self.assay = None
        self.age_unit = None
        self.max_age = -1
        self.min_age = -1

        super().__init__(specimen_id)

    def get_dict(self):
        specimen_dict = super().get_dict()

        specimen_dict["DataProperties"]["SPR.hasAgeUnit"] = self.age_unit
        specimen_dict["DataProperties"]["SPR.hasMaxAge"] = self.max_age
        specimen_dict["DataProperties"]["SPR.hasMinAge"] = self.min_age

        specimen_dict["AnnotationProperties"]["SR.hasSpecimenID"] = self.specimen_ID
        specimen_dict["AnnotationProperties"]["SR.hasAssayID"] = self.assay

        return specimen_dict
