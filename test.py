from TestingOrchestrators.TestingOrchestrator import TestingOrchestrator
from Models.BenchmarkType import BenchmarkType

def main():
      TestingOrchestrator.NAIMAValidator4x(BenchmarkType.NYUV2)
      TestingOrchestrator.NAIMAValidator8x(BenchmarkType.NYUV2)
      TestingOrchestrator.NAIMAValidator16x(BenchmarkType.NYUV2)

      TestingOrchestrator.NAIMAValidator4x(BenchmarkType.RGBDD)
      TestingOrchestrator.NAIMAValidator8x(BenchmarkType.RGBDD)
      TestingOrchestrator.NAIMAValidator16x(BenchmarkType.RGBDD)

      TestingOrchestrator.NAIMAValidator4x(BenchmarkType.TOFDSRD)
      TestingOrchestrator.NAIMAValidator8x(BenchmarkType.TOFDSRD)
      TestingOrchestrator.NAIMAValidator16x(BenchmarkType.TOFDSRD)

      TestingOrchestrator.NAIMAValidatorBenchamrk4x(BenchmarkType.MIDDLE)
      TestingOrchestrator.NAIMAValidatorBenchamrk8x(BenchmarkType.MIDDLE)
      TestingOrchestrator.NAIMAValidatorBenchamrk16x(BenchmarkType.MIDDLE)

      TestingOrchestrator.NAIMAValidatorBenchamrk4x(BenchmarkType.LU)
      TestingOrchestrator.NAIMAValidatorBenchamrk8x(BenchmarkType.LU)
      TestingOrchestrator.NAIMAValidatorBenchamrk16x(BenchmarkType.LU)

if __name__ == '__main__':
    main()