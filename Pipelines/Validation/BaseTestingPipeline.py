from Configurations.ValidationConfigurations import ValidationConfigurations
from Configurations.ValidationDataConfigurations import ValidationDataConfigurations
from Models.BenchmarkType import BenchmarkType
from Pipelines.PipelineBase import PipelineBase
from Utilities.PathManager import PathManager
import torch.nn as nn
import torch
import os

class BaseTestingPipeline(PipelineBase):
    def __init__(
        self,
        valid_data_path: str = PathManager.GetBasePath() + BenchmarkType.NYUV2.name,
        model_load_path: str = PathManager.GetBasePath() + 'model_states',
        model_name: str = 'last.pth',
        total_example: int = 100,
        eval_scale: int = 4,
        patch_size_valid: int = None, # Does not matter
    ):
        self._valid_data_path = valid_data_path
        self._valid_data_pathScale = ''
        self._model_load_path = model_load_path
        self._model_name = model_name
        self._total_example = total_example
        self._eval_scale = eval_scale
        self._patch_size_valid = patch_size_valid


    def InitModel(self, model: nn.Module):
        self.model = model

    def LoadConfigurations(self,):
        self.configurations = ValidationConfigurations(
            model_path=os.path.join(self._model_load_path, self._model_name),
            data_configurations=ValidationDataConfigurations(
                patch_size=self._patch_size_valid,
                augment=False,
                batch_size=1,
                base_folder=self._valid_data_path,
                repeat=1,
                scale=self._eval_scale,
                total_examples=self._total_example,
                eval_scale=self._eval_scale,
            ),
            save_path=self._model_load_path
        )

    def CreateDataLoaders(self,):
        pass

    def LoadModelWeights(self, ):
        self.saved_model = torch.load(self.configurations.model_path)
        self.model.load_state_dict(self.saved_model['model'])
        # self.model = torch.compile(self.model)

    def InitTrainingRecipe(self, ):
        pass

    def InitModelObjectives(self, ):
        pass