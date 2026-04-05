from Models.BenchmarkType import BenchmarkType
from Models.ModelType import ModelType
from Trainers.NAIMATrainer import NAIMATrainer

class TrainingOrchestrator:
    BATCH = 3 # Batch 1 for 16 scale
    REPEAT = 4 # Repeat 8 for 16 scale
    SCALE = 4
    MODEL = ModelType.NAIMA
    BENCHMARK = BenchmarkType.NYUV2

    @staticmethod
    def Train():
        print('Batch, Repeat, Scale, Model, Benchmark: ', TrainingOrchestrator.BATCH, TrainingOrchestrator.REPEAT, TrainingOrchestrator.SCALE, TrainingOrchestrator.MODEL, TrainingOrchestrator.BENCHMARK)
    
        if TrainingOrchestrator.MODEL == ModelType.NAIMA:
            NAIMATrainer(
                model=TrainingOrchestrator.MODEL,
                benchmark_type=TrainingOrchestrator.BENCHMARK,
                input_patch=448 if TrainingOrchestrator.SCALE >= 8 else 420,
                scale=TrainingOrchestrator.SCALE,
                repeat=TrainingOrchestrator.REPEAT,
                batch_size=TrainingOrchestrator.BATCH,
            ).TrainModel()
        else:
            raise Exception('Specified model not implemented.')

    