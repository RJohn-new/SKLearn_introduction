# Robert Johnson

# Import statements
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier as Mlp
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sn

# All the class labels
classes = ['cp', 'im', 'pp', 'imU', 'om', 'omL', 'imL', 'imS']


# Read CSV File with column names and drop the STR column
def read_data():
    columns = ['Seqn', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm2', 'alm1', 'class']
    return pd.read_csv('data/ecoli.csv', header=None, names=columns).drop('Seqn', axis=1)


# Split data into x and y, then into test and train
def split_data(df):
    x = df.iloc[:, : -1].values
    y = df.iloc[:, -1].values
    return train_test_split(x, y, test_size=0.2, random_state=1)


# Classification with Decision Tree
def use_tree(train_x, test_x, train_y, test_y):
    dec_tree = DecisionTreeClassifier()
    dec_tree = dec_tree.fit(train_x, train_y)
    predictions = dec_tree.predict(test_x)
    conf_mat = metrics.confusion_matrix(test_y, predictions, labels=classes)
    acc = metrics.accuracy_score(test_y, predictions)
    precision = metrics.precision_score(test_y, predictions, average='weighted', zero_division=0)
    recall = metrics.recall_score(test_y, predictions, average='weighted', zero_division=0)
    return conf_mat, acc, precision, recall


# Classification with naive bayes
def use_naive_bayes(train_x, test_x, train_y, test_y):
    model = GaussianNB()
    model.fit(train_x, train_y)
    preds = model.predict(test_x)
    conf_mat = metrics.confusion_matrix(test_y, preds, labels=classes)
    acc = metrics.accuracy_score(test_y, preds)
    precision = metrics.precision_score(test_y, preds, average='weighted', zero_division=0)
    recall = metrics.recall_score(test_y, preds, average='weighted', zero_division=0)
    return conf_mat, acc, precision, recall


# Classification with Neural Network
def use_nn(train_x, test_x, train_y, test_y):
    sc = StandardScaler()
    train_x, test_x = sc.fit_transform(train_x), sc.fit_transform(test_x)
    model = Mlp(hidden_layer_sizes=6, batch_size=1, max_iter=10000, activation='identity', solver='lbfgs')
    model.fit(train_x, train_y)
    preds = model.predict(test_x)
    conf_mat = metrics.confusion_matrix(test_y, preds, labels=classes)
    acc = metrics.accuracy_score(test_y, preds)
    precision = metrics.precision_score(test_y, preds, average='weighted', zero_division=0)
    recall = metrics.recall_score(test_y, preds, average='weighted', zero_division=0)
    return conf_mat, acc, precision, recall


# Classification with Support Vector Machine
def use_svm(train_x, test_x, train_y, test_y):
    model = svm.SVC()
    model.fit(train_x, train_y)
    preds = model.predict(test_x)
    conf_mat = metrics.confusion_matrix(test_y, preds, labels=classes)
    acc = metrics.accuracy_score(test_y, preds)
    precision = metrics.precision_score(test_y, preds, average='weighted', zero_division=0)
    recall = metrics.recall_score(test_y, preds, average='weighted', zero_division=0)
    return conf_mat, acc, precision, recall


# Display all four confusion matrices as heatmaps
def show_matrices(matrices):
    mats = ['Support Vector Machine', 'Neural Network', 'Naive Bayes', 'Decision Tree']
    index = 0
    for mat in matrices:
        df_mat = pd.DataFrame(mat)
        plt.figure(figsize=(8, 6))
        plt.suptitle('Confusion Matrix ' + mats[index])
        index += 1
        sn.set(font_scale=1.4)
        sn.heatmap(df_mat, annot=True, annot_kws={'size': 16}, fmt='d')
    plt.show()


# Bar graph of the accuracies
def graph_accuracy(accuracies):
    labels = ['D Tree', 'Naive Bayes', 'Neural Net', 'SVM']
    l_pos = np.arange(len(labels))

    accuracies = [x * 100 for x in accuracies]

    plt.bar(l_pos, accuracies, align='center')
    plt.xticks(l_pos, labels)
    plt.yticks(np.arange(0, 100, step=10))
    plt.ylabel('Accuracy')
    plt.title('Accuracy Percentages by Algorithm')
    plt.show()


# Bar graph of the precisions
def graph_precision(precisions):
    labels = ['D Tree', 'Naive Bayes', 'Neural Net', 'SVM']
    l_pos = np.arange(len(labels))

    precisions = [x * 100 for x in precisions]

    plt.bar(l_pos, precisions, align='center')
    plt.xticks(l_pos, labels)
    plt.yticks(np.arange(0, 100, step=10))
    plt.ylabel('Precision')
    plt.title('Precision Percentages by Algorithm')
    plt.show()


# Bar graph of the recalls
def graph_recall(recalls):
    labels = ['D Tree', 'Naive Bayes', 'Neural Net', 'SVM']
    l_pos = np.arange(len(labels))

    recalls = [x * 100 for x in recalls]

    plt.bar(l_pos, recalls, align='center')
    plt.xticks(l_pos, labels)
    plt.yticks(np.arange(0, 100, step=10))
    plt.ylabel('Recall')
    plt.title('Recall Percentages by Algorithm')
    plt.show()


# Main Method
def main():
    df = read_data()
    train_x, test_x, train_y, test_y = split_data(df)

    tree_mat, tree_acc, tree_prec, tree_rec = use_tree(train_x, test_x, train_y, test_y)
    nb_mat, nb_acc, nb_prec, nb_rec = use_naive_bayes(train_x, test_x, train_y, test_y)
    nn_mat, nn_acc, nn_prec, nn_rec = use_nn(train_x, test_x, train_y, test_y)
    svm_mat, svm_acc, svm_prec, svm_rec = use_svm(train_x, test_x, train_y, test_y)

    show_matrices((svm_mat, nn_mat, nb_mat, tree_mat))
    graph_accuracy((tree_acc, nb_acc, nn_acc, svm_acc))
    graph_precision((tree_prec, nb_prec, nn_prec, svm_prec))
    graph_recall((tree_rec, nb_rec, nn_rec, svm_rec))


# Launch Main method
if __name__ == '__main__':
    main()
