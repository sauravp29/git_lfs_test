

class SchemaDictLoadError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class SchemaDictSegregationError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class Stage1FeatureCreationError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class Stage2FeatureCreationError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class Stage3FeatureCreationError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class FeatureRenameError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class FeatureConcatenationError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class FloatConversionError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message
