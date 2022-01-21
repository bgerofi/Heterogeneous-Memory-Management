#/usr/bin/python
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import hashlib

if len(sys.argv) != 5:
	print("error: specify input file, application, type and output file name")
	sys.exit(1)

#sns.set_theme()
sns.set_style("whitegrid")

data = pd.read_csv(sys.argv[1], sep=",", skipinitialspace=True)

data = data[(data["App"] == sys.argv[2]) & (data["Type"] == sys.argv[3])]
print(data)
if len(data) == 0:
    printf("error: no data selected?")
    sys.exit(1)

if sys.argv[3] == "Perc":
    plt.figure(figsize=(6,5))
    plt.xticks(size=11)
else:
    plt.figure(figsize=(6,6))
    plt.xticks(rotation=85, size=10)
    for i in data.index:
        if data.at[i, "Layout"].startswith("DRAM+HBM-0x"):
            data.at[i, "Layout"] = "DRAM+HBM-" + hashlib.md5(data.at[i, "Layout"].encode()).hexdigest()[0:6]

# Convert to seconds
for i in data.index:
    data.at[i, "Runtime"] /= 1000

#plt.xticks(size=11)
plt.yticks(size=11)

ax = sns.barplot(x="Layout", y="Runtime", data=data, hue="Datapoint", capsize=.1);
ax.plot(np.nan, 'xr', label = 'Error')
ax.plot(np.nan, 'xb', label = 'Error (Phasemarked)')

ax.set_ylabel('Runtime (sec)', size=14)
ax2 = ax.twinx()
palette = {"Estimated":"tab:red",
           "Estimated (Phasemarked)":"tab:blue",
           "Measured":"tab:blue"}
sns.scatterplot(x="Layout", y="Error", ax=ax2, data=data, hue="Datapoint", palette=palette, s=100, marker="x", legend=False);
#handlesr, labelsr = ax2.get_legend_handles_labels()
#ax.legend(handles=(handles[0:] + handlesr[0:]), labels=(labels[0:] + labelsr[0:]))

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=(handles[2:] + handles[0:2]), labels=(labels[2:] + labels[0:2]))

ax2.grid(False)
ax2.set_ylabel('Error (%)', size=14)
ax2.set_ylim(0, data["Error"].max() * 1.5)
if sys.argv[3] == "Perc":
    ax.set_xlabel('Percentage placed into HBM', size=14)
    ax2.set_xlabel('Percentage placed into HBM', size=14)
else:
    ax.set_xlabel('Memory layout', size=14)
#plt.ylim(0.9, 2)
plt.tight_layout()

#plt.setp(ax.get_legend().get_texts(), fontsize='8') # for legend text
#plt.setp(ax.get_legend().get_title(), fontsize='9')

plt.savefig(sys.argv[4])
