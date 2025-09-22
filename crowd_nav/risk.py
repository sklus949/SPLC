import pickle
import math

def risk():
    with open('data/danger.pkl', 'rb') as f:
        data1 = pickle.load(f)
        sorted_data1 = sorted(data1)
    len1 = len(data1)
    print(len1)
    min1 = sorted_data1[int(len1 * 0.05)]
    max1 = sorted_data1[int(len1 * 0.95)]
    print(f"min1: {min1}, max1: {max1}")


    with open('data/human_labels.pkl', 'rb') as f:
        data3 = pickle.load(f)
        # print(data3)

    cnt = 0
    for i in range(len(data3)):
        data3[i] = data3[i] - 0.1 * math.tanh((data1[cnt] - data1[cnt + 1]) / (max1 - min1))
        cnt += 2

    with open('data/human_labels.pkl', 'wb') as f:
        pickle.dump(data3, f)

    with open('data/human_labels.pkl', 'rb') as f:
        data3 = pickle.load(f)
        # print(data3)