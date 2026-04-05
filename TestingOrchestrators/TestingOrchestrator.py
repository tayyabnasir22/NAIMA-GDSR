from Models.BenchmarkType import BenchmarkType
from Models.ModelType import ModelType
from ValidationHelpers.NAIMA_ValidationHelper import NAIMA_ValidationHelper
from ValidationHelpers.NAIMA_ValidationHelperBenchmark import NAIMA_ValidationHelperBenchmark
from Validators.NAIMA_Validator import NAIMA_Validator

class TestingOrchestrator:
    @staticmethod
    def NAIMAValidator16x(data = BenchmarkType):
        print('Running ' + data.value + ', Scale: ', str(16), 'x')
        validator = NAIMA_Validator(
            model=ModelType.NAIMA, 
            validation_helper=NAIMA_ValidationHelper(16), 
            benchmark_type=data
        )
        validator.TestModel('./' + data.value, './model_states_NAIMA_NYUV2_Patch_448_Scale_16', 'last.pth', 449, 16)
        print('-'*33)

    @staticmethod
    def NAIMAValidator8x(data = BenchmarkType):
        print('Running ' + data.value + ', Scale: ', str(8), 'x')
        validator = NAIMA_Validator(
            model=ModelType.NAIMA, 
            validation_helper=NAIMA_ValidationHelper(8), 
            benchmark_type=data
        )
        validator.TestModel('./' + data.value, './model_states_NAIMA_NYUV2_Patch_448_Scale_8', 'last.pth', 449, 8)
        print('-'*33)

    @staticmethod
    def NAIMAValidator4x(data = BenchmarkType):
        print('Running ' + data.value + ', Scale: ', str(4), 'x')
        validator = NAIMA_Validator(
            model=ModelType.NAIMA, 
            validation_helper=NAIMA_ValidationHelper(4), 
            benchmark_type=data
        )
        validator.TestModel('./' + data.value, './model_states_NAIMA_NYUV2_Patch_420_Scale_4', 'last.pth', 449, 4)
        print('-'*33)

    @staticmethod
    def NAIMAValidatorBenchamrk16x(data = BenchmarkType):
        print('Running ' + data.value + ', Scale: ', str(16), 'x')
        validator = NAIMA_Validator(
            model=ModelType.NAIMA, 
            validation_helper=NAIMA_ValidationHelperBenchmark(16), 
            benchmark_type=data
        )
        validator.TestModel('./' + data.value, './model_states_NAIMA_NYUV2_Patch_448_Scale_16', 'last.pth', 449, 16)
        print('-'*33)

    @staticmethod
    def NAIMAValidatorBenchamrk8x(data = BenchmarkType):
        print('Running ' + data.value + ', Scale: ', str(8), 'x')
        validator = NAIMA_Validator(
            model=ModelType.NAIMA, 
            validation_helper=NAIMA_ValidationHelperBenchmark(8), 
            benchmark_type=data
        )
        validator.TestModel('./' + data.value, './model_states_NAIMA_NYUV2_Patch_448_Scale_8', 'last.pth', 449, 8)
        print('-'*33)

    @staticmethod
    def NAIMAValidatorBenchamrk4x(data = BenchmarkType):
        print('Running ' + data.value + ', Scale: ', str(4), 'x')
        validator = NAIMA_Validator(
            model=ModelType.NAIMA, 
            validation_helper=NAIMA_ValidationHelperBenchmark(4), 
            benchmark_type=data
        )
        validator.TestModel('./' + data.value, './model_states_NAIMA_NYUV2_Patch_420_Scale_4', 'last.pth', 449, 4)
        print('-'*33)
