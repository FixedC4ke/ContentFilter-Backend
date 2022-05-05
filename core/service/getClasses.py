from joblib import load
import pathlib
import sys
import json
import numpy
baseDir = pathlib.Path(__file__).parent.parent

le = load(baseDir.joinpath('model', 'le.joblib'))

print(json.dumps({"categories": numpy.array(le.classes_).tolist()}))

sys.stdout.flush()