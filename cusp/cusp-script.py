import subprocess

matrices = open("matrices_620.txt", 'r')
writeTimes = open("times.txt", 'w')
writeFeatures = open("features.txt", 'w')

finalList = []
for line in matrices:
    name = line.rstrip().split(' ')
    directory = "/home/matrices/mtx/" + name[0] + "/" + name[1] + "/" + name[1] + ".mtx"
    result = [0, 0, 0]
    tests = [0, 1, 3] #COO, CSR, ELL
    features = subprocess.check_output(["./spmv", "5", directory])
    featureList = features.split(' ')
    resultFeatureString = name[1] + features + "\n"
    if features[12]/features[4] <= 3:
        for i in range(0, 3):
            result[i] = subprocess.check_output(["./spmv", str(tests[i]), directory])
    else:
        for i in range(0, 2):
            result[i] = subprocess.check_output(["./spmv", str(tests[i]), directory])
        result[2] = "N/A"
    resultTimeString = name[1] + " " + str(result[0]) + " " + str(result[1]) + " " + str(result[2]) + "\n"
    writeTimes.write(resultTimeString)
    writeFeatures.write(resultFeatureString)


