from ModelFactories.BaseModelFactory import BaseModelFactory
from Models.BenchmarkType import BenchmarkType
from Models.ModelType import ModelType
from Models.RunningAverage import RunningAverage
from Models.Timer import Timer
from Pipelines.Validation.BaseNAIMATestingPipeline import BaseNAIMATestingPipeline
from Pipelines.Validation.BaseNAIMATestingPipelineBenchmark import BaseNAIMATestingPipelineBenchmark
from Utilities.Logger import Logger
from ValidationHelpers.NAIMA_ValidationHelper import NAIMA_ValidationHelper
from Validators.BaseValidator import BaseValidator
import os

class NAIMA_Validator(BaseValidator):
    def __init__(
            self, 
            model: ModelType, 
            validation_helper: NAIMA_ValidationHelper,
            benchmark_type: BenchmarkType):
        model = model
        super().__init__(
            model=model,
            validation_helper=validation_helper,
            benchmark_type=benchmark_type,
        )
        

    def _RunTests(
            self, 
            pipeline: BaseNAIMATestingPipeline, 
            factory: BaseModelFactory, 
        ):
        img_size = 448 if self._validation_helper.scale > 4 else 420
        factory.BuildModel(pipeline, self._model, use_pretrained = False, scale=self._validation_helper.scale, patch_size = 14, img_size=img_size)

        # 4. Call the testing
        timer = Timer()
        results: dict[str, RunningAverage] = self._GetPrediction(pipeline)

        # 5. Pass the result to writer
        out_path = Logger.LogTestResultsToCSV(
            results, 
            self._benchmark_type, 
            pipeline.configurations.data_configurations.eval_scale, 
            os.path.dirname(pipeline.configurations.model_path), 
            timer, 
            pipeline.configurations.model_path, 
        )

        Logger.Log('Test results saved at: ', out_path)

    def TestModel(self, valid_data_path, model_load_path, model_name, total_example, eval_scale):
        # 1. Init thre required Pipeline
        print(valid_data_path, model_load_path)
        if self._benchmark_type in [BenchmarkType.LU, BenchmarkType.MIDDLE]:
            pipeline = BaseNAIMATestingPipelineBenchmark(
                 valid_data_path=valid_data_path,
                model_load_path=model_load_path,
                model_name=model_name,
                total_example=total_example,
                eval_scale=eval_scale,
                patch_size_valid=None,
            )
        else:
            pipeline = BaseNAIMATestingPipeline(
                valid_data_path=valid_data_path,
                model_load_path=model_load_path,
                model_name=model_name,
                total_example=total_example,
                eval_scale=eval_scale,
                patch_size_valid=None,
            )

        # 2. Build the model
        factory = BaseModelFactory()
        
        self._RunTests(pipeline, factory)
