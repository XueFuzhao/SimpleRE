import json
import numpy as np
import argparse

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def getresult(fn):
    result = []
    with open(fn, "r") as f:
        l = f.readline()
        while l:
            l = l.strip().split()
            for i in range(len(l)):
                l[i] = float(l[i])
            result += [l]
            l = f.readline()
    result = np.asarray(result)
    return list(1 / (1 + np.exp(-result)))

def getpredict(result, T1 = 0.5, T2 = 0.4):
    for i in range(len(result)):
        r = []
        maxl, maxj = -1, -1
        for j in range(len(result[i])):
            if result[i][j] > T1:
                r += [j]
            if result[i][j] > maxl:
                maxl = result[i][j]
                maxj = j
        if len(r) == 0:
            if maxl <= T2:
                r = [36]
            else:
                r += [maxj]
        result[i] = r
    return result

def evaluate(devp, data):
    index = 0
    correct_sys, all_sys = 0, 0
    correct_gt = 0
    
    for i in range(len(data)):
        for j in range(len(data[i][1])):
            for id in data[i][1][j]["rid"]:
                if id != 36:
                    correct_gt += 1
                    if id in devp[index]:
                        correct_sys += 1
            for id in devp[index]:
                if id != 36:
                    all_sys += 1
            index += 1

    precision = correct_sys/all_sys if all_sys != 0 else 1
    recall = correct_sys/correct_gt if correct_gt != 0 else 0
    f_1 = 2*precision*recall/(precision+recall) if precision+recall != 0 else 0

    return precision, recall, f_1



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--f1dev",
                        default=None,
                        type=str,
                        required=True,
                        help="Dev logits (f1).")
    parser.add_argument("--f1test",
                        default=None,
                        type=str,
                        required=True,
                        help="Test logits (f1).")

    
    args = parser.parse_args()
    
    f1dev = args.f1dev
    f1test = args.f1test


    with open("data/dev.json", "r", encoding='utf8') as f:
        datadev = json.load(f)
    with open("data/test.json", "r", encoding='utf8') as f:
        datatest = json.load(f)
    for i in range(len(datadev)):
        for j in range(len(datadev[i][1])):
            for k in range(len(datadev[i][1][j]["rid"])):
                datadev[i][1][j]["rid"][k] -= 1
    for i in range(len(datatest)):
        for j in range(len(datatest[i][1])):
            for k in range(len(datatest[i][1][j]["rid"])):
                datatest[i][1][j]["rid"][k] -= 1

    bestT2 = bestf_1 = 0
    for T2 in range(51):
        dev = getresult(f1dev)
        devp = getpredict(dev, T2=T2/100.)
        precision, recall, f_1 = evaluate(devp, datadev)
        if f_1 > bestf_1:
            bestf_1 = f_1
            bestT2 = T2/100.

    print("best T2:", bestT2)

    dev = getresult(f1dev)
    devp = getpredict(dev, T2=bestT2)
    precision, recall, f_1 = evaluate(devp, datadev)
    print("dev (P R F1)", precision, recall, f_1)

    test = getresult(f1test)
    testp = getpredict(test, T2=bestT2)
    precision, recall, f_1 = evaluate(testp, datatest)
    print("test (P R F1)", precision, recall, f_1)

    
