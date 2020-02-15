import json
from glob import glob
import numpy as np


if __name__ == "__main__":
    base_path = "../output/ecg/"
    for name in ["baseline", "kd", "small"]:
        files = glob(base_path + name + "*")
        f1 = [json.load(open(x))["f1"] for x in files]
        mean_f1 = np.mean(f1)
        std_f1 = np.std(f1)
        print(name, "f1", "%s +- %s" % (mean_f1, std_f1))
