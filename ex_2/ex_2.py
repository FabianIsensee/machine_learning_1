import numpy as np
from sklearn import datasets
from sklearn import cross_validation
import IPython

# uses variances to select features. we select the features where the total variance is larger than the variance within
# each class.
def dr(x, y):
    total_var = np.var(x, axis=0)
    class_var = np.zeros(total_var.shape)
    for c in np.unique(y):
        class_var += np.var(x[y == c, :], axis = 0) * np.sum(y == c) / float(len(y))
    return x[:, np.argsort(total_var - class_var)[::-1][:2]]

def nm(trainingx, trainingy, testx):
    mean_vec = []
    class_correspondence = np.unique(trainingy)
    for c in np.unique(trainingy):
        mean_vec += [[np.mean(trainingx[trainingy == c, :][:, i]) for i in range(trainingx.shape[1])]]

    # im too lazy to make this fast. double for loop it is
    def dist(query, class_mean):
        return np.sqrt(np.sum(np.square(query-class_mean)))

    pred = np.zeros(testx.shape[0])
    for i in range(testx.shape[0]):
        distances = []
        for j in range(len(mean_vec)):
            distances += [dist(testx[i], mean_vec[j])]
        pred[i] = class_correspondence[np.argmin(distances)]
    return pred

def compute_qda(trainingy, trainingx):
    mean_vec = []
    cov_matrices = []
    priors = []
    class_correspondence = np.unique(trainingy)
    for c in np.unique(trainingy):
        my_mean = [np.mean(trainingx[trainingy == c, :][:, i]) for i in range(trainingx.shape[1])]
        mean_vec += [my_mean]
        my_data = trainingx[trainingy == c, :]

        my_data_nomean = np.array([my_data[:, i] - my_mean[i] for i in range(my_data.shape[1])]).transpose()

        cov_matrices += [1. / float(np.sum(trainingy == c)) * np.dot(my_data_nomean.transpose(), my_data_nomean)]
        priors += [np.sum(trainingy == c) / float(len(trainingy))]
    return mean_vec, cov_matrices, priors

def perform_qda(mu, cov, priors, testx):
    n_classes = len(mu)
    assert len(mu) == len(cov)
    assert len(cov) == len(priors)
    pred = np.ones(testx.shape[0])*999
    inv_cov = [np.linalg.inv(i) for i in cov]

    for i in xrange(pred.shape[0]):
        scores = []
        for j in xrange(n_classes):
            b_k = np.log(np.linalg.det(2. * np.pi * cov[j])) - 2. * np.log(priors[j])
            x_prime = testx[i]
            this_score = b_k + np.dot(np.dot((x_prime - mu[j]).transpose(), inv_cov[j]), (x_prime - mu[j]))
            scores += [this_score]
        pred[i] = np.argmin(scores)
    return pred



if __name__ == "__main__":
    # load data
    digits = datasets.load_digits()

    data = digits.data
    target = digits.target

    # keep only 1's and 7's
    data = data[(target == 1) | (target == 7), :]
    target = target[(target == 1) | (target == 7)]

    data_reduced = dr(data, target)

    assert data_reduced.shape[0] == 361

    # split into train and test set
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(data_reduced, target, test_size=0.333, random_state=0)

    # scatterplot
    import matplotlib.pyplot as plt
    plt.title('Scatter Plot')
    plt.xlabel('X1')
    plt.ylabel('X2')

    size = 12

    plt.scatter(X_train[:, 0][Y_train == 1], X_train[:, 1][Y_train == 1], marker='x', c='r', s=size)
    plt.scatter(X_train[:, 0][Y_train == 7], X_train[:, 1][Y_train == 7], marker='o', c='b', s=size)
    plt.show() # looks good

    # nearest mean
    testy = nm(X_train, Y_train, X_test)
    testy_accur = np.sum(Y_test==testy)/float(len(Y_test))
    print "Nearest Mean Classifier: %f accuracy on test set" % testy_accur

    # train qda
    mu, cov, prior = compute_qda(Y_train, X_train)

    # predict with qda
    pred_qda = perform_qda(mu, cov, prior, X_test)
    IPython.embed()
    class_correspondence = np.unique(Y_train)
    pred_qda = class_correspondence[pred_qda.astype("int")]
    pred_qda_accur = np.sum(Y_test==pred_qda)/float(len(Y_test))

    print "QDA: %f accuracy on test set" % pred_qda_accur

    # visualize decision boundary
    feat_0_lower_lim = np.min(X_train[:, 0])
    feat_1_lower_lim = np.min(X_train[:, 1])
    feat_0_upper_lim = np.max(X_train[:, 0])
    feat_1_upper_lim = np.max(X_train[:, 1])

    visualize_train_x = np.zeros((100 * 100, 2))
    for i in xrange(100):
        for j in xrange(100):
            visualize_train_x[i * 100 + j] = np.array([feat_0_lower_lim + 1./100. * float(i) * float(feat_0_upper_lim),
                                                       feat_1_lower_lim + 1./100. * float(j) * float(feat_1_upper_lim)])

    # predict visualize_train_x
    pred_visualize_data = perform_qda(mu, cov, prior, visualize_train_x)
    pred_visualize_data = class_correspondence[pred_visualize_data.astype("int")]

    plt.title('Scatter Plot - QDA decision boundary')
    plt.xlabel('X1')
    plt.ylabel('X2')

    size = 12

    plt.scatter(visualize_train_x[:, 0][pred_visualize_data == 1], visualize_train_x[:, 1][pred_visualize_data == 1], marker='x', c='r', s=size)
    plt.scatter(visualize_train_x[:, 0][pred_visualize_data == 7], visualize_train_x[:, 1][pred_visualize_data == 7], marker='o', c='b', s=size)
    plt.show() # looks good




