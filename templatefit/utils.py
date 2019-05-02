import os
import numpy as np

def load_template_file(filename):
    """ """
    if not os.path.exists(filename):
        raise IOError("File does not exist: %s"%filename)

    templates_data = np.genfromtxt(filename, delimiter=",", names=True)

    line_names = templates_data.dtype.names

    priors = templates_data['prior']
    try:
        ntempl = len(priors)
    except TypeError:
        ntempl = 1
        priors = np.array([priors])

    return templates_data, priors