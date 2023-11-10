
class DictLoadingFailedException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return 'DictLoadingFailedException : ' + str(self.message)

    def __repr__(self) -> str:
        return 'DictLoadingFailedException : ' + str(self.message)



class RawDataPreprocessFailedException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return 'RawDataPreprocessFailedException : ' + str(self.message)

    def __repr__(self) -> str:
        return 'RawDataPreprocessFailedException : ' + str(self.message)
    


class ClusteringDFSFeatureCreationFailedException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return 'ClusteringDFSFeatureCreationFailedException : ' + str(self.message)

    def __repr__(self) -> str:
        return 'ClusteringDFSFeatureCreationFailedException : ' + str(self.message)
    


class DFSFeatureCreationFailedException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return 'DFSFeatureCreationFailedException : ' + str(self.message)

    def __repr__(self) -> str:
        return 'DFSFeatureCreationFailedException : ' + str(self.message)
    


class ChunkingDataFailedException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return 'ChunkingDataFailedException : ' + str(self.message)

    def __repr__(self) -> str:
        return 'ChunkingDataFailedException : ' + str(self.message)
    


class FeatureCreationFailedException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return 'FeatureCreationFailedException : ' + str(self.message)

    def __repr__(self) -> str:
        return 'FeatureCreationFailedException : ' + str(self.message)


class SchemaDictCreationFailedException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return 'SchemaDictCreationFailedException : ' + str(self.message)

    def __repr__(self) -> str:
        return 'SchemaDictCreationFailedException : ' + str(self.message)
    



class GetDefaultFeaturesFailedException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return 'GetDefaultFeaturesFailedException : ' + str(self.message)

    def __repr__(self) -> str:
        return 'GetDefaultFeaturesFailedException : ' + str(self.message)
    


class LoadingDataFailedException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return 'LoadingDataFailedException : ' + str(self.message)

    def __repr__(self) -> str:
        return 'LoadingDataFailedException : ' + str(self.message)
    


class RenameFormatDataFailedException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return 'RenameFormatDataFailedException : ' + str(self.message)

    def __repr__(self) -> str:
        return 'RenameFormatDataFailedException : ' + str(self.message)



