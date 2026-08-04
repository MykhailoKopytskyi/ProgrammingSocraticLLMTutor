from .common.config import SAMPLE_CASES
from .simulator.simulator import Simulator

for case in SAMPLE_CASES:
    simulator = Simulator(case)
    results = simulator.run()
    print(results)
