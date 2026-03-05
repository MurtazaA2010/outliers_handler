import sys
from pathlib import Path

# Add parent directory to path to find anml module
sys.path.insert(0, str(Path(__file__).parent.parent))

from AnLOF.AnLOF_module import AnLOF
from sample_data import X_train, X_val, y_train, y_val, features

anlof = AnLOF(X_train, X_val, y_train, y_val, features)

methods = [
    anlof.IQR,
    anlof.z_score,
    anlof.winsorize,
    anlof.median_method,
    anlof.mean_method,
    anlof.robust_scaler,
    anlof.standard_scaler,
    anlof.minmax_scaler,
    anlof.quantile_normal,
    anlof.log_transform,
]

for method in methods:
    X_tr, X_vl = method()
    print(method.__name__, "✅ works")
