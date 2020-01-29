import csv
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from operator import itemgetter


path_in = r"/home/irene/PycharmProjects/04_Risk_Model/data/NL_all_with_comments_RD_New_ready.csv"

dicenv = defaultdict(int)
dicact = defaultdict(int)

with open(path_in, "r", newline="") as r:
    reader = csv.reader(r, delimiter=";")
    header = next(reader)
    for row in reader:
        env = row[-5]
        act = row[-4]

        if "-" in env:
            lenv = env.split("-")
        else:
            lenv = [env]

        if "-" in act:
            lact = act.split("-")
        else:
            lact = [act]

        for item in lenv:
            dicenv[item] += 1

        for item in lact:
            dicact[item] += 1


lin = np.linspace(0, 10, 11)

le = [(key, dicenv[key]) for key in sorted(dicenv.keys())]
la = [(key, dicact[key]) for key in sorted(dicact.keys())]

le_sort = list(reversed(sorted(le, key=itemgetter(1))))
la_sort = list(reversed(sorted(la, key=itemgetter(1))))

env_lbl = [item[0] for item in le_sort]
env_dat = [item[1] for item in le_sort]

act_lbl = [item[0] for item in la_sort]
act_dat = [item[1] for item in la_sort]

plt.subplot(1, 2, 1)
plt.grid(zorder=0)
plt.title("Type of environment reported by volunteers", size=24)
plt.bar(lin, env_dat, tick_label=env_lbl)
plt.xticks(lin, env_lbl, fontsize=20, rotation=70)
plt.yticks(fontsize=20)
plt.ylim(0, 26000)

plt.subplot(1, 2, 2)
plt.grid(zorder=0)
plt.title("Type of activity reported by volunteers", size=24)
plt.bar(lin, act_dat, tick_label=act_lbl)
plt.xticks(lin, act_lbl, fontsize=20, rotation=70)
plt.yticks(fontsize=20)
plt.ylim(0, 26000)

plt.show()


