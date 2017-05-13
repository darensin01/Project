import pandas as pd
import numpy as np
import time
import random
import multiprocessing
#from functools import partial

from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

STORE_NAME = 'DataStore.h5'
LABEL_KEY = 'shuffledLabels'
DATA_KEY = 'shuffledData'
NUM_PROCESSES = 4

# Genetic algorithm parameters
K = 100
CROSSOVER = 0.5
MUTATION = 0.05
ELITISM = 0.1
RUNNER_UP = 0.1
NUM_GENERATIONS = 30
POP_SIZE = 50
SOLN_SIZE = 2000
ELIMINATION = 0.5
THRESHOLD_PERCENT = 0.01
THRESHOLD_GEN = 5

# Get data from store.
store = pd.HDFStore(STORE_NAME)
shuffledData = pd.read_hdf(store, DATA_KEY)
labels = (pd.read_hdf(store, LABEL_KEY)).values
store.close()

def compute_rqn(values):

    (cases, features) = shuffledData.shape

    # Calculate probability of P(Y=1) and P(Y=0).
    prob_label_1 = np.mean(labels)
    prob_label_0 = 1 - prob_label_1

    # Find lists of **indices** where the labels are 0 or 1.
    index_label_1 = [j for j in xrange(cases) if labels[j] == 1]
    index_label_0 = [j for j in xrange(cases) if labels[j] == 0]

    # Return a list of results for the process working on "values".
    results = []

    # Iterate through the interval for which each process is responsible.
    for i in values:
        feature = shuffledData.iloc[:, i]

        # E(X ^ 2).
        exp_feature_sq = np.mean(np.square(feature))

        # E(X | Y = y).
        exp_feature_label_1 = np.mean(feature[index_label_1])
        exp_feature_label_0 = np.mean(feature[index_label_0])

        # var(X). Returns unbiased estimate of variance with ddof = 1.
        var_feature = np.var(feature, ddof=1)

        # E(var(X | Y))
        cond_exp = exp_feature_sq - (prob_label_1 * exp_feature_label_1 ** 2) \
               - (prob_label_0 * exp_feature_label_0 ** 2)

        # Compute correlation coefficient.
        results.append((i, 1 - cond_exp / var_feature))

    return results


def partition(features):
    # Returns a list of list of indices that are partitioned according to
    # 'total', which is the number of processes.
    # Optional selected_idx parameter. If this is specified, then we remove
    # the already selected indices from the candidate indices.

    interval = features / NUM_PROCESSES
    return [range(index * interval, min((index + 1) * interval, features))
                    for index in range(NUM_PROCESSES + 1)]


def create_random_bit_array(N, k):
    values = [0] * N

    # Picks randomly chosen **indices** within the N-sized array "values".
    # Ensures that the indices that are randomly chosen are unique.
    random_unique_ints = set()
    while len(random_unique_ints) < k:
        random_unique_ints.add(random.randint(0, N))

    # Makes "values" a binary-encoded array, where 1 represents an index is
    # chosen based on the selection from the above uniform distribution.
    for i in range(N):
        if i in random_unique_ints:
            values[i] += 1

    return values


def mutate(solution):
    num_mutated_bits = int(MUTATION * len(solution))

    # Check that the array still contains K 1's. The variable iterations ensures
    # that we enter the while loop, since solution already sums to K.
    iterations = 0
    while sum(solution) != K or iterations == 0:
        # Select which indices to mutate. Ensures indices to mutate are unique.
        indices_mutate_unique = set()
        while len(indices_mutate_unique) < num_mutated_bits:
            indices_mutate_unique.add(random.randint(0, len(solution)))

        # Flip the bits.
        for i in range(len(solution)):
            if i in indices_mutate_unique:
                solution[i] = flip_bit(solution[i])

        iterations = 1

    return solution


def flip_bit(x):
    if x == 1:
        return 0

    if x == 0:
        return 1


def crossover(parents):
    # "parents" is the binary encoding of features, not indices.
    parent_one = parents[0]
    parent_two = parents[1]

    # Goes through each bit in the parent solution. Then, sample uniformly from
    # a uniform distribution. If this number exceeds the CROSSOVER threshold,
    # then we swap the ith bit of both parents. Changes both parents in place.
    for i in range(len(parent_one)):
        sample = np.random.uniform(0.0, 1.0)
        if sample >= CROSSOVER:
            temp = parent_one[i]
            parent_one[i] = parent_two[i]
            parent_two[i] = temp

    return [parent_one, parent_two]


if __name__ == '__main__':

    (cases, features) = shuffledData.shape

    # Compute correlation coefficient (rqn) between each feature and target
    # labels. Partition function returns a list of list of indices. Map the
    # compute_rqn function into each list of indices for each process to work
    # on.
    start_time = time.time()
    pool = multiprocessing.Pool(processes = NUM_PROCESSES)
    results = pool.map(compute_rqn, partition(features))

    # Flatten lists of list into a single list.
    index_rqns = [item for sublist in results for item in sublist]

    # Return list of sorted tuples (index, rqn score). Sorted according to
    # rqn decreasing order of the scores.
    index_rqn = sorted(index_rqns, key=lambda(idx, score):score, reverse=True)

    # Select first SOLN_SIZE number of candidates from index_rqn list. This is
    # the list that the genetic algorithm starts off with. Will contain the
    # **indices** of the candidate features. For example, if SOLN_SIZE = 2000,
    # we select the top 2000 best (idx, score) pairs from index_rqn.
    initial_candidates = index_rqn[:SOLN_SIZE]

    # Conduct GA until fitness does not increase THRESHOLD_PERCENT for
    # THRESHOLD_GEN generations. Thus, we have to keep track of the best fitness
    # so far. We also need a "final solution" that the GA outputs.
    # consecutive_gen keeps track of the number of consecutive generations that
    # the fitness does not improve much.
    best_fitness = 0
    consecutive_gen = 0
    num_iterations = 1
    final_solution = []
    while consecutive_gen < THRESHOLD_GEN:

        print "Current iteration: %i" % num_iterations

        # Creates a list of size POP_SIZE of randomly generated binary
        # indicators. random_bits is a list of 0's and 1's of length SOLN_SIZE,
        # which indicate which index in initial_candidates is "on" or "off".
        generation_solution = []
        generation_cv_scores = []
        for i in range(POP_SIZE):
            random_bits = create_random_bit_array(SOLN_SIZE, K)
            generation_solution.append(random_bits)

            # Create an array which stores only the **indices** for which
            # random_bits is 1.
            selected_indices = [j for j in range(SOLN_SIZE) if random_bits[j] == 1]

            # Create an array which stores the actual indices (of shuffledData).
            selected_features = [initial_candidates[j] for j in selected_indices]

            # Evaluate fitness. Fit into linear SVM and compute cross validation
            # score.
            selected_feature_values = shuffledData.iloc[:, selected_features]
            selected_feature_values = map(preprocessing.scale, selected_feature_values)
            clf = SVC(kernel='linear')
            cv_scores = cross_val_score(clf, selected_feature_values, y=labels, cv=5, scoring='accuracy')
            generation_cv_scores.append((i, np.mean(cv_scores)))

        # We have a list of the ith solution in generation_solution and its
        # corresponding score stored in generation_cv_scores. Sort this list
        # from highest fitness to lowest fitness.
        sorted_scores = sorted(generation_cv_scores, key=lambda(pop_idx, score):score, reverse=True)

        # Keep track of this generation's best fitness.
        current_best_fitness = sorted_scores[0][1]

        # Keep track of the next generation's solutions.
        next_gen_soln = []

        # Pick best solutions from sorted_scores. Clone best performing
        # solutions for next generation. top_solutions stores only the indices
        # of generation_solution.
        num_elitism = int(POP_SIZE * ELITISM)
        top_solutions_idx = map(lambda(pop_idx, score):pop_idx, sorted_scores[:num_elitism])

        # Creates a list of top solutions. Append this list to next_gen_soln.
        top_solutions = [generation_solution[i] for i in top_solutions_idx]
        next_gen_soln.append(top_solutions)

        # Remove those top_solutions_idx which have already been selected due to
        # elitism. Select indices for mutation.
        del sorted_scores[:num_elitism]
        num_mutation = int(POP_SIZE * RUNNER_UP)
        mutation_idx = map(lambda(pop_idx, score):pop_idx, sorted_scores[:num_mutation])
        mutation_soln = [generation_solution[i] for i in mutation_idx]

        # Mutate each of the selected solution in mutation_soln.
        # Also checks that the number of 1's in each solution is maintained.
        for soln in mutation_soln:
            next_gen_soln.append(mutate(soln))

        # Remove those mutation_idx which have already been selected due to
        # mutation. Remove (from the back of list) also weakest solutions.
        del sorted_scores[:num_mutation]
        del sorted_scores[-int((ELIMINATION * POP_SIZE)):]

        # Use Roulette Wheel selection to select parents to generate offsprings.
        # Randomly select parents from remaining indices. Normalise scores first
        # to produce a list of probabilities. Parents chosen with probability
        # proportional to its score.
        sum_scores = sum(map(lambda(idx, score):score, sorted_scores))
        probabilities = [(idx, score / sum_scores) for (idx, score) in sorted_scores]

        # Need to ensure that next_gen_soln has the correct number of solutions.
        while len(next_gen_soln) < POP_SIZE:
            # Choose two parents, with replacement, based on probabilities.
            parents = np.random.choice(sorted_scores, size=2, replace=True, p=probabilities)
            parent_idx = map(lambda(idx, score):idx, parents)
            parent_soln = [generation_solution[i] for i in parent_idx]

            # Conduct crossover on the two parent solutions. Returns two
            # children.
            offsprings = crossover(parent_soln)

            # Conduct mutation on the two children. Also helps to ensure that
            # both children have the correct number of 1's.
            for child in offsprings:
                next_gen_soln.append(mutate(child))

        # Check if best_fitness is not increasing much relative to the
        # previous iteration. If the best_fitness increases by a significant
        # amount again, reset consecutive_gen back to 0.
        if best_fitness > 0 and (current_best_fitness - best_fitness) / best_fitness <= THRESHOLD_PERCENT:
            consecutive_gen += 1
        else:
            consecutive_gen = 0

        best_fitness = current_best_fitness
        final_solution = next_gen_soln

    # We have final solution presented by the GA algorithm. Evaluate again the
    # final set of next_gen_soln in terms of cross-validation error.
    final_scores = []
    for i in range(POP_SIZE):
        selected_indices = [j for j in range(SOLN_SIZE) if final_solution[j] == 1]

        # Create an array which stores the actual indices (of shuffledData).
        selected_features = [initial_candidates[j] for j in selected_indices]

        # Evaluate fitness. Fit into linear SVM and compute cross validation
        # score.
        selected_feature_values = shuffledData.iloc[:, selected_features]
        selected_feature_values = map(preprocessing.scale, selected_feature_values)
        clf = SVC(kernel='linear')
        cv_scores = cross_val_score(clf, selected_feature_values, y=labels, cv=5, scoring='accuracy')
        final_scores.append((i, np.mean(cv_scores)))

    final_scores = sorted(final_scores, lambda(idx, score):score, reverse=True)
    best_index = final_scores[0][0]
    best_indices = final_solution[best_index]
    best_score = final_scores[0][1]
    best_feature_indices = [initial_candidates[i] for i in best_indices]

    end_time = time.time()

    # Write results to a text file.
    filename = 'ga_results.txt'
    f = open(filename, 'w')
    f.write("The best score is %f" % best_score)
    for i in range(len(best_feature_indices)):
        f.write("The best indices are...")
        f.write("%i" % (best_feature_indices[i]))
        f.write("\n")

    f.write('Time taken: %f' % (end_time - start_time))
    f.write("\n")
    f.close()



