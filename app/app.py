from .common.config import CODE_MISCONCEPTIONS
from .simulator.simulator import Simulator

for misc in CODE_MISCONCEPTIONS:
    simulator: Simulator = Simulator(misc)
    results = simulator.run()
    print(results)
