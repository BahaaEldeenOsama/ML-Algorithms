import numpy as np
import pandas as pd


class TreeNode:
    def __init__(self, lbl):
        self.label = lbl
        self.nxt = {}

    def predict(self, x):
        if x[self.label] not in self.nxt:
            return 'unpredictable'

        go_next = self.nxt[x[self.label]]
        if not isinstance(go_next, TreeNode):
            return go_next
        return go_next.predict(x)


def split_to_train_test(df, train_ratio):
    train = df.sample(frac=train_ratio)
    test = df.drop(train.index)
    return train, test


def calculate_entropy(y):
    classes, count = np.unique(y, return_counts=True)
    size = float(len(y))

    entropy_value = np.sum(
        [(-count[i] / size) * np.log2(count[i] / size) for i in range(len(classes))]
    )
    return entropy_value


def calculate_information_gain(feat, y):
    children, details = np.unique(feat, return_inverse=True)

    avg = 0
    for j in range(len(children)):
        child = y[details == j]
        entropy_child = calculate_entropy(child)

        avg += entropy_child * len(child) / len(y)

    return calculate_entropy(y) - avg


def recurse(x, y, features):
    unique, cnt = np.unique(y, return_counts=True)
    if len(unique) <= 1:
        return unique[0], 1
    if len(features) == 0:
        return unique[np.argmax(cnt)], 1

    gains = [calculate_information_gain(x[feat], y) for feat in features]
    optimal = features[np.argmax(gains)]
    node = TreeNode(optimal)
    leaves_size = 0
    for choice in np.unique(x[optimal]):
        subset = x[optimal] == choice
        node.nxt[choice], tmp_leave_size = recurse(x[subset], y[subset], [v for v in features if v != optimal])
        leaves_size += tmp_leave_size

    return node, (leaves_size + 1)


def generate_decision_tree(x, y):
    return recurse(x, y, features=x.columns.tolist())


# Fills missing data points by majority of the row
def fill_in_unknowns(df, majority):
    x = df.iloc[:, 1:]
    for i in range(x.shape[1]):
        x.iloc[:, i].replace('?', 'y' if majority[i] else 'n', inplace=True)


def find_majority(df):
    x = df.iloc[:, 1:]
    return (x.isin(['y']).sum(axis=0) >= x.isin(['n']).sum(axis=0)).tolist()


def calculate_accuracy(tree, df):
    total = 0
    for i in range(len(df)):
        if tree.predict(df.iloc[i, 1:]) == df.iloc[i, 0]:
            total += 1
    return total / (len(df) * 1.0)

def testing():
    train = pd.read_csv('part1_data/lecture.txt', header=None)
    tree = generate_decision_tree(train.iloc[:, 1:], train.iloc[:, 0])
    print("Training accuracy: ", calculate_accuracy(tree, train))

def exercising(df, train_ratio, iterations, fill_in=False):
    train_accuracies = []
    test_accuracies = []
    tree_sizes = []

    for i in range(iterations):
        train, test = split_to_train_test(df, train_ratio)
        if fill_in:
            majority = find_majority(train)
            fill_in_unknowns(train, majority)
            fill_in_unknowns(test, majority)
        tree, tree_size = generate_decision_tree(train.iloc[:, 1:], train.iloc[:, 0])

        train_accuracies.append(calculate_accuracy(tree, train))
        test_accuracies.append(calculate_accuracy(tree, test))
        tree_sizes.append(tree_size)

        print("Training accuracy(" + str(train_ratio) + "): " + str(train_accuracies[i]))
        print("Testing  accuracy(" + str(1 - train_ratio) + "): " + str(test_accuracies[i]))
        print("Tree size: " + str(tree_size))

    return train_accuracies, test_accuracies, tree_sizes


def main():
    df = pd.read_csv('house-votes-84.data.txt', header=None)
    iterations = 5
    ratios = [0.25, 0.30, 0.40, 0.50, 0.60, 0.70]
    ratios_size = len(ratios)

    print("Run 5 times: training Ratios(0.25, 0.30, 0.40, 0.50, 0.60,0.07)\n")
    for i in range(ratios_size):
        print("For Train Ratio(" + str(ratios[i]) + ")\n")
        train_accuracies, test_accuracies, tree_sizes = exercising(df, ratios[i], iterations, fill_in=False if i == 0 else True)

        print("\nMin Train accuracy: " + str(np.min(train_accuracies)) +
              "\nMax Train accuracy: " + str(np.max(train_accuracies)) +
              "\nMean Train accuracy: " + str(np.mean(train_accuracies)))

        print("Min Test accuracy: " + str(np.min(test_accuracies)) +
              "\nMax Test accuracy: " + str(np.max(test_accuracies)) +
              "\nMean Test accuracy: " + str(np.mean(test_accuracies)))

        print("Min Tree size: " + str(np.min(tree_sizes)) +
              "\nMax Tree size: " + str(np.max(tree_sizes)) +
              "\nMean Tree size: " + str(np.mean(tree_sizes)))
        print("---------------------------------------------------")


if __name__ == '__main__':
    main()