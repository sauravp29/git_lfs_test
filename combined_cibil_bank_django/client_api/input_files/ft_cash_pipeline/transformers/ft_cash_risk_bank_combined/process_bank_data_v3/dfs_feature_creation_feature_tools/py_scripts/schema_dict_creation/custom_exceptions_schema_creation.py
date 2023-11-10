class SchemaDictSortingStageError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class SchemaDictFeatureResolvingError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message
