from sklearn.neural_network import MLPRegressor
import numpy as np

csr_data = open("timesCSR_2017-07-25_10_46_54.863055.txt", 'r')
ell_data = open("timesELL_2017-07-25_10_46_54.863055.txt", 'r')
coo_data = open("timesCOO_2017-07-25_10_46_54.863055.txt", 'r')
feature_data = open("features_2017-07-25_10_46_54.863055.txt", 'r')
output = open("mlp-scoring.txt", 'w')

csr_times = []
coo_times = []
ell_times = []
ell_times_final = []
n = []
nnz = []
dis = []
mu = []
sd = []
nmax = []
ell_n = []
ell_nnz = []
ell_dis = []
ell_nmax = []
names = []
n_count = 0

# Data Parsing
for line in feature_data:
    features = line.rstrip().split(' ')
    names.append(features[0] + " - " + features[1])
    n.append(float(features[3]))
    nnz.append(float(features[5]))
    dis.append(float(features[7]))
    mu.append(float(features[9]))
    sd.append(float(features[11]))
    nmax.append(float(features[13]))

for line in csr_data:
    time = line.rstrip().split(' ')
    csr_times.append(float(time[2]))
for line in coo_data:
    time = line.rstrip().split(' ')
    coo_times.append(float(time[2]))
for line in ell_data:
    time = line.rstrip().split(' ')
    if time[2] == 'N/A':
        ell_times.append("N/A")
    else:
        ell_times.append(float(time[2]))
    n_count += 1
for i in range(0, len(ell_times)):
    if ell_times[i] != 'N/A':
        ell_times_final.append(ell_times[i])
        ell_n.append(n[i])
        ell_nnz.append(nnz[i])
        ell_dis.append(dis[i])
        ell_nmax.append(nmax[i])

ell_X = []
coo_X = []
csr_X = []

for i in range(0, len(n)):
    coo_X.append([n[i], nnz[i], dis[i]])
    csr_X.append([n[i], nnz[i], dis[i], mu[i], sd[i]])
for i in range(0, len(ell_n)):
    ell_X.append([ell_n[i], ell_nnz[i], ell_dis[i], ell_nmax[i]])

### Preprocessing COO X
coo_X_log = np.log10(np.array(coo_X))
mean_X = np.mean(coo_X_log,axis = 0)
std_X = np.std(coo_X_log, axis=0)
coo_X_norm = np.multiply((coo_X_log - mean_X), (1 / std_X))

### Preprocessing COO Y
coo_log_times = np.log10(np.array(coo_times))
coo_mean_Y = np.mean(coo_log_times, axis = 0)
coo_std_Y = np.std(coo_log_times, axis = 0)
coo_norm_times = np.multiply((coo_log_times - coo_mean_Y), (1 / coo_std_Y))

### Preprocessing ELL X
ell_X_log = np.log10(np.array(ell_X))
mean_X = np.mean(ell_X_log,axis = 0)
std_X = np.std(ell_X_log, axis=0)
ell_X_norm = np.multiply((ell_X_log - mean_X), (1 / std_X))

### Preprocessing ELL Y
ell_log_times = np.log10(np.array(ell_times_final))
ell_mean_Y = np.mean(ell_log_times, axis = 0)
ell_std_Y = np.std(ell_log_times, axis = 0)
ell_norm_times = np.multiply((ell_log_times - ell_mean_Y), (1 / ell_std_Y))

### Preprocessing CSR X
csr_eps = np.add(csr_X, 1)
csr_X_log = np.log10(np.array(csr_eps))
csr_mean_X = np.mean(csr_X_log, axis=0)
csr_std_X = np.std(csr_X_log, axis=0)
csr_X_norm = np.multiply((csr_X_log - csr_mean_X), (1 / csr_std_X))

### Preprocessing CSR Y
csr_log_times = np.log10(np.array(csr_times))
csr_mean_Y = np.mean(csr_log_times, axis = 0)
csr_std_Y = np.std(csr_log_times, axis = 0)
csr_norm_times = np.multiply((csr_log_times - csr_mean_Y), (1 / csr_std_Y))

train_coo_X = []
test_coo_X = []
train_coo_Y = []
test_coo_Y = []

train_csr_X = []
test_csr_X = []
train_csr_Y = []
test_csr_Y = []

train_ell_X = []
test_ell_X = []
train_ell_Y = []
test_ell_Y = []

### Split
for i in range(0, 10):
    test_coo_X.append(coo_X_norm[62*i: 62*(i+1)])
    x_temp = np.concatenate((coo_X_norm[0: 62*i], coo_X_norm[62*(i+1): 620]), axis=0)
    train_coo_X.append(x_temp)
    test_coo_Y.append(coo_norm_times[62 * i: 62 * (i + 1)])
    y_temp = np.concatenate((coo_norm_times[0: 62 * i], coo_norm_times[62 * (i + 1): 620]), axis=0)
    train_coo_Y.append(y_temp)
    x_temp = []
    y_temp = []

for i in range(0, 10):
    test_csr_X.append(csr_X_norm[62 * i: 62 * (i + 1)])
    x_temp = np.concatenate((csr_X_norm[0: 62 * i], csr_X_norm[62 * (i + 1): 620]), axis=0)
    train_csr_X.append(x_temp)
    test_csr_Y.append(csr_norm_times[62 * i: 62 * (i + 1)])
    y_temp = np.concatenate((csr_norm_times[0: 62 * i], csr_norm_times[62 * (i + 1): 620]), axis=0)
    train_csr_Y.append(y_temp)
    x_temp = []
    y_temp = []

for i in range(0, 10):
    test_ell_X.append(ell_X_norm[20*i: 20*(i+1)])
    x_temp = np.concatenate((ell_X_norm[0: 20*i], ell_X_norm[20*(i+1): 200]), axis=0)
    train_ell_X.append(x_temp)
    test_ell_Y.append(ell_norm_times[20 * i: 20 * (i + 1)])
    y_temp = np.concatenate((ell_norm_times[0: 20 * i], ell_norm_times[20 * (i + 1): 200]), axis=0)
    train_ell_Y.append(y_temp)
    x_temp = []
    y_temp = []

output.write("COO" + "\n")
for test in range(0, 10):
    clf_coo = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(len(coo_X[0]), 1), random_state=1,
                        learning_rate_init=0.01, momentum=0.2)
    coo_model = clf_coo.fit(train_coo_X[test], train_coo_Y[test])
    #print(coo_model)
    prediction_coo = coo_model.predict(test_coo_X[test])
    print "score of coo testing set " + str(test) + ": " + str(clf_coo.score(test_coo_X[test], test_coo_Y[test]))
    coo_mult = np.multiply(prediction_coo, coo_std_Y)
    coo_sum_mean = coo_mult + coo_mean_Y
    coo_pred = np.power(10, coo_sum_mean)

    coo_y_data = coo_times[62 * test: 62 * (test + 1)]

    sum_coo = 0
    for i in range(0, len(test_coo_X[test])):
        # print str(prediction_coo[i]) + " vs " + str(coo_y_test[i])
        # print str(coo_pred[i]) + " vs " + str(coo_y_data[i])
        # if abs((coo_pred[i] - coo_y_data[i]) / coo_y_data[i]) > 0.5:
        # print abs((coo_pred[i] - coo_y_data[i]) / coo_y_data[i])
        sum_coo += abs((coo_pred[i] - coo_y_data[i]) / coo_y_data[i])
    print "rme of coo: " + str(sum_coo / len(test_coo_X[test]))
    output.write(str(test) + " " + str(clf_coo.score(test_coo_X[test], test_coo_Y[test])) + " " +
                 str(sum_coo / len(test_coo_X[test])) + "\n")

output.write("CSR" + "\n")
for test in range(0, 10):
    clf_csr = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(len(csr_X[0]), 1), random_state=1,
                        learning_rate_init=0.01, momentum=0.2)
    csr_model = clf_csr.fit(train_csr_X[test], train_csr_Y[test])
    #print(csr_model)
    prediction_csr = csr_model.predict(test_csr_X[test])
    print "score of csr testing set " + str(test) + ": " + str(clf_csr.score(test_csr_X[test], test_csr_Y[test]))
    csr_mult = np.multiply(prediction_csr, csr_std_Y)
    csr_sum_mean = csr_mult + csr_mean_Y
    csr_pred = np.power(10, csr_sum_mean)

    csr_y_data = csr_times[62 * test: 62 * (test + 1)]

    sum_csr = 0
    for i in range(0, len(test_csr_X[test])):
        # print str(prediction_csr[i]) + " vs " + str(csr_y_test[i])
        # print str(csr_pred[i]) + " vs " + str(csr_y_data[i])
        # if abs((csr_pred[i] - csr_y_data[i]) / csr_y_data[i]) > 0.5:
        # print abs((csr_pred[i] - csr_y_data[i]) / csr_y_data[i])
        sum_csr += abs((csr_pred[i] - csr_y_data[i]) / csr_y_data[i])
    print "rme of csr: " + str(sum_csr / len(test_csr_X[test]))
    output.write(str(test) + " " + str(clf_csr.score(test_csr_X[test], test_csr_Y[test])) + " " +
                 str(sum_csr / len(test_csr_X[test])) + "\n")

output.write("ELL" + "\n")
for test in range(0, 10):
    clf_ell = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(len(ell_X[0]), 1), random_state=1,
                        learning_rate_init=0.01, momentum=0.2)
    ell_model = clf_ell.fit(train_ell_X[test], train_ell_Y[test])
    #print(ell_model)
    prediction_ell = ell_model.predict(test_ell_X[test])
    print "score of ell testing set " + str(test) + ": " + str(clf_ell.score(test_ell_X[test], test_ell_Y[test]))
    ell_mult = np.multiply(prediction_ell, ell_std_Y)
    ell_sum_mean = ell_mult + ell_mean_Y
    ell_pred = np.power(10, ell_sum_mean)

    ell_y_data = ell_times_final[20 * test: 20 * (test + 1)]

    sum_ell = 0
    for i in range(0, len(test_ell_X[test])):
        # print str(prediction_ell[i]) + " vs " + str(ell_y_test[i])
        # print str(ell_pred[i]) + " vs " + str(ell_y_data[i])
        # if abs((ell_pred[i] - ell_y_data[i]) / ell_y_data[i]) > 0.5:
        # print abs((ell_pred[i] - ell_y_data[i]) / ell_y_data[i])
        sum_ell += abs((ell_pred[i] - ell_y_data[i]) / ell_y_data[i])
    print "rme of ell: " + str(sum_ell / len(test_ell_X[test]))
    output.write(str(test) + " " + str(clf_ell.score(test_ell_X[test], test_ell_Y[test])) + " " +
                 str(sum_ell / len(test_ell_X[test])) + "\n")