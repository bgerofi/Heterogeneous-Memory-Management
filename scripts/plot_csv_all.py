#/usr/bin/python
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

if len(sys.argv) != 3:
	print("error: specify input file and output file name")
	sys.exit(1)

sns.set_theme()
#sns.set_style("whitegrid")

data = pd.read_csv(sys.argv[1], sep=",", skipinitialspace=True)
#data = data.set_index("Nodes")
#data = data.astype({"cycles": float})

print(data)

#plt.figure(figsize=(10,20))
plt.xticks(rotation=80, size=8)
#plt.xticks(size=11)
plt.yticks(size=11)
#plt.rcParams["figure.figsize"] = (9, 4)

ax = sns.catplot(x="Memory", y="Runtime", hue="Datapoint", col="App", data=data, kind="bar");

#ax.set_ylabel('Runtime (ms)', size=14)
#ax.set_xlabel('Memory device', size=14)
#plt.ylim(0.9, 2)
plt.tight_layout()

#plt.setp(ax.get_legend().get_texts(), fontsize='8') # for legend text
#plt.setp(ax.get_legend().get_title(), fontsize='9')

plt.savefig(sys.argv[2])
