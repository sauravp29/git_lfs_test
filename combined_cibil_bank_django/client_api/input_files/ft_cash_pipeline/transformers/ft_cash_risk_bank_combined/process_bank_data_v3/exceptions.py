class RawJsonEmpty(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class RawBankingPreprocessFailedException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class RawBankingNoDataAfterFilter(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class TextClassificationFailedException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class SplitSMSDebitCreditFailedException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class CreateFeatNonDebitCreditSMSFailedException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class SplitDebitCreditSMSForDFSFailedException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class CreatePrimAndSecAccountsFailedException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class CreateNonDFSFeatFailedException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class SchemaDictLoadError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class HolidayDFLoadingFailedException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class TextClassificationModelLoadingFailedException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class JsonExtractionDataFailedException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class AdditionalColumnCreationFailedException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class FractionFeatureCreationFaield(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class NachClassificationFailedException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class PipelineFourDataPrepocException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class MappingFileLoadingFailedException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class MappingFileNotFoundError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class DataFrameRenamingException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class MultiplecreatedDateException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message
# class MultiplecreatedDateException(Exception):
#     def __init__(self, message):
#         super().__init__(message)
#         self.message = message
