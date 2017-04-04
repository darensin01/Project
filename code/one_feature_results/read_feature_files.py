import pandas as pd

filename = "all_one_feature_results.txt"
f = open(filename, 'a')

count = 0
scoreIndex = []

for i in range(0, 11):
    print i

    name = "one_feature_selection_results_" + str(i) + ".txt"
    with open(name) as fs:
        content = fs.readlines()
        for x in range(0, len(content)):

            if x == len(content) - 1:
                print content[x]
                break

            score_index_array = content[x].strip().split(";")
            score_str = score_index_array[0]
            index_str = score_index_array[1]

            score = float(score_str.split(":")[1])
            index = int(index_str.split(":")[1])
            index_score = (index, score)
            scoreIndex.append(index_score)
            count += 1

print count

# Sort the list based on score.
sorted_list = sorted(scoreIndex, key=lambda (index, score):score, reverse=True)

sorted_indices = map(lambda (index, score):index, sorted_list)
sorted_scores = map(lambda (index, score):score, sorted_list)

results_series = pd.Series(sorted_scores, index=sorted_indices)

for pair in sorted_list:
    f.write(str(pair) + "\n")

f.close()

STORE_NAME = 'DataStore.h5'
store = pd.HDFStore(STORE_NAME)
results_series.to_hdf(store, 'sorted_one_feature_scores')

store.close()

