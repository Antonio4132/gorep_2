from abc import ABC, abstractmethod
from Project import Project


def flat_list(items):
    if type(items) is not list:
        return items

    items2 = []
    for item in items:
        if type(item) is list:
            items2 = items2 + item
        else:
            items2 = items2 + [item]

    return items2


class OntologyConversorAbstract(ABC):

    def __init__(self):
        self.specimen = None
        self.project = None
        self.mapping_dict = self.init_map()

        super().__init__()

    def format_specimen(self, raw_specimen, specimen_id):
        self.format_concrete_specimen(raw_specimen, specimen_id)

        return self.specimen

    def format_project(self, raw_project, project_id):
        self.format_concrete_project(raw_project, project_id)

        return self.project

    def map_word(self, word):
        # If word is a list, apply map_word for each item
        if type(word) is list:
            return flat_list(list(map(self.map_word, word)))
        try:
            word_mapped = self.mapping_dict[word]
            return flat_list(word_mapped)
        except KeyError:
            return word

    def parse_word(self, word):
        # If word is None, return None
        if word is None:
            return None

        # If text is a list, apply parse_word for each item
        if type(word) is list:
            return flat_list(list(map(self.parse_word, word)))

        # Parse word depending on the repository
        word_parsed = self.parse_concrete(word)

        # Map word if necessary
        return flat_list(self.map_word(word_parsed))

    ####################################################
    # region Abstract methods
    @abstractmethod
    def init_map(self):
        pass

    @abstractmethod
    def format_concrete_specimen(self, raw_specimen, specimen_id):
        pass

    @abstractmethod
    def format_concrete_project(self, raw_project, project_id):
        pass

    @abstractmethod
    def parse_concrete(self, word):
        pass

    # endregion
    ####################################################
