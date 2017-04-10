import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from ast import literal_eval as make_tuple

params = {'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large',
          'legend.fontsize': 20}
pylab.rcParams.update(params)

filename = 'all_results.txt'

f = open(filename, 'w')

indices = []
scores = []

for i in range(1, 101):
    with open('rfs_results/all_' + str(i) + '_feature_results.txt') as fs:
        content = fs.readline()
        f.write(content)
        index_score = make_tuple(content)
        indices.append(index_score[0])
        scores.append(index_score[1])

f.close()

rfs, = plt.plot(range(1, 101), scores, 'ko', label="RFS")
full, = plt.plot(range(0, 101), [0.834644] * 101, 'k--', linewidth=3, label="All features")
plt.legend(handles=[rfs, full], loc=4)
plt.xlabel("Number of features", fontsize=25)
plt.ylabel("Cross validation score", fontsize=25)
plt.title("Cross validation score against number of features for RFS using SVM", fontsize=25)
plt.show()

print scores.index(max(scores))
