"""
Name: Reynaldo Gomez
Last Edited: 2/18/2026

Description:

This script loads the LSWMD wafer map dataset from a pickle file that was created using an older version of pandas and would otherwise fail to open in a modern environment due to 
deprecated module paths. It resolves this by remapping the legacy pandas index modules so the dataset can be reconstructed correctly during loading.

The code inspects its structure and processes the failureType column to extract usable scalar labels from nested array formats that prevent direct analysis. 
It then identifies the unique defect classes and computes their frequency counts, enabling an initial assessment of class imbalance across the labeled wafer defect categories for downstream CNN training.

Notes:
From the LSWMD.pkl dataset; we know that:
    - 811,457 total rows but 638,507 are unlabeled; this leaves 172,950 usable labeled rows for training.
    - 9 classes: 8 defect types + (no defect)
    - Imbalance with 147k samples but `Near-full` only has 149 so WeightedRandomSampler will be crucial

"""


import sys
import types
import pickle as pkl
from pathlib import Path
import pandas
import numpy as np
# These three submodules still exist in pandas 2.x under core.indexes
# We import them directly so we can remap them below
import pandas.core.indexes.base
import pandas.core.indexes.range
import pandas.core.indexes.multi


"""
 How pickle loading works:
   When pkl.load() encounters a class reference, it does:
       sys.modules[module_name].class_name
   If the module doesn't exist in sys.modules, it raises ModuleNotFoundError.
 
 FIX:
   Manually insert fake module objects into sys.modules under the OLD paths,
   pointing to the NEW locations. Pickle finds the old name, gets redirected
   to the modern equivalent, and reconstructs the object successfully.
"""


pandas_indexes = types.ModuleType("pandas.indexes")
pandas_indexes.base    = pandas.core.indexes.base  #type: ignore
pandas_indexes.range   = pandas.core.indexes.range #type: ignore
pandas_indexes.multi   = pandas.core.indexes.multi #type: ignore
pandas_indexes.numeric = pandas.core.indexes.base  #type: ignore

# Register all old paths in sys.modules so pickle's import machinery finds them
sys.modules["pandas.indexes"]         = pandas_indexes
sys.modules["pandas.indexes.base"]    = pandas.core.indexes.base
sys.modules["pandas.indexes.range"]   = pandas.core.indexes.range
sys.modules["pandas.indexes.multi"]   = pandas.core.indexes.multi
sys.modules["pandas.indexes.numeric"] = pandas.core.indexes.base

pkl_path = Path("Dataset/LSWMD.pkl")
with pkl_path.open("rb") as f:
    obj = pkl.load(f, encoding="latin-1")

print("Shape:", obj.shape)
print("Columns:", obj.columns.tolist())
print("\nFirst row sample:")
print(obj.iloc[0])

# Safe unique — flatten nested arrays before deduplicating
def extract_label(val):
    """
    This is a numpy array containing a list containing the label string.
    Calling pandas .unique() on this fails with "unhashable type: numpy.ndarray"
    because arrays can't be used as dict keys in the hash table unique() uses.

    This function peels off nesting layers until it reaches the scalar string.
    Empty arrays (unlabeled wafers) become the string '[]'.

    """

    while isinstance(val, (list, np.ndarray)) and len(val) > 0:
        val = val[0]
    return str(val)

labels = obj["failureType"].apply(extract_label)
print("\nfailureType unique values:", sorted(labels.unique()))
print("\nfailureType value counts:")
print(labels.value_counts())