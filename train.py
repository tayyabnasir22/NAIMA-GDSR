from Models.BenchmarkType import BenchmarkType
from Models.ModelType import ModelType
from TrainingOrchestrators.TrainingOrchestrator import TrainingOrchestrator
import sys
from Utilities.PathManager import PathManager

def main(scale, model):
    PathManager.BASE_PATH = './'

    models = {
        'v': ModelType.NAIMA
    }

    TrainingOrchestrator.SCALE = scale
    TrainingOrchestrator.MODEL = models[model]
    TrainingOrchestrator.BENCHMARK = BenchmarkType.NYUV2

    TrainingOrchestrator.REPEAT = 2
    TrainingOrchestrator.BATCH = 1
    
    TrainingOrchestrator.Train()

if __name__ == '__main__':
    args_len = len(sys.argv) - 1

    main(int(sys.argv[1]), sys.argv[2])