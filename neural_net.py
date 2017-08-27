from sklearn.neural_network import MLPRegressor
import numpy as np

csr_data = open("timesCSR_2017-07-25_10_46_54.863055.txt", 'r')
ell_data = open("timesELL_2017-07-25_10_46_54.863055.txt", 'r')
coo_data = open("timesCOO_2017-07-25_10_46_54.863055.txt", 'r')
feature_data = open("features_2017-07-25_10_46_54.863055.txt", 'r')

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

print csr_X_norm
print csr_norm_times

### Split
coo_X_train = np.array(coo_X_norm[62:])
coo_X_test = np.array(coo_X_norm[:62])

csr_X_train = np.array(csr_X_norm[62:])
csr_X_test = np.array(csr_X_norm[:62])

ell_X_train = np.array(ell_X_norm[62:])
ell_X_test = np.array(ell_X_norm[:62])

coo_Y_train = np.array(coo_norm_times[62:])
coo_Y_test = np.array(coo_norm_times[:62])

csr_Y_train = np.array(csr_norm_times[62:])
csr_Y_test = np.array(csr_norm_times[:62])

ell_Y_train = np.array(ell_norm_times[62:])
ell_Y_test = np.array(ell_norm_times[:62])

clf_coo = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(len(coo_X[0]), 1), random_state=1,
                        learning_rate_init=0.01, momentum=0.2)
clf_csr = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(len(csr_X[0]), 1), random_state=1,
                        learning_rate_init=0.01, momentum=0.2)
clf_ell = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(len(ell_X[0]), 1), random_state=1,
                        learning_rate_init=0.01, momentum=0.2)

#print clf_coo
print clf_csr
#print clf_ell

clf_coo.fit(coo_X_train, coo_Y_train)
clf_csr.fit(csr_X_train, csr_Y_train)
clf_ell.fit(ell_X_train, ell_Y_train)

#print clf_coo.predict(coo_X_test)
print clf_csr.predict(csr_X_test)
#print clf_ell.predict(ell_X_test)

#print clf_coo.score(coo_X_test, coo_Y_test)
print clf_csr.score(csr_X_test, csr_Y_test)
#print clf_ell.score(ell_X_test, ell_Y_test)

