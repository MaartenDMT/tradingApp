from model.rl_system.factory import RLAlgorithmFactory

f = RLAlgorithmFactory()
f.initialize()
print(f'Available algorithms: {f.get_available_algorithms()}')
print(f'Total count: {len(f.get_available_algorithms())}')
