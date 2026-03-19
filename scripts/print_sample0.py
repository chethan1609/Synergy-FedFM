import importlib.machinery, importlib.util, sys, json, numpy as np, os
lhs_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'lhs_sweep.py'))
loader = importlib.machinery.SourceFileLoader('lhs_sweep', lhs_path)
spec = importlib.util.spec_from_loader(loader.name, loader)
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), '..')))
lhs = importlib.util.module_from_spec(spec)
loader.exec_module(lhs)
param_ranges={
    'quantize_bits': (8,32,True),
    'M_max': (100,1000,True),
    'r_k': (4,16,True),
    'gamma': (0.001,0.05,False),
    'lambda_reg': (0.0,0.2,False)
}
rng = np.random.default_rng(seed=12345)
samples = lhs.lhs_sampling(12,param_ranges,rng)
import numpy as _np
def to_py(o):
    if isinstance(o, _np.integer):
        return int(o)
    if isinstance(o, _np.floating):
        return float(o)
    return o

clean = {k: to_py(v) for k,v in samples[0].items()}
print(json.dumps(clean, indent=2))
