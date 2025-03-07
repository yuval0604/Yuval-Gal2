import numpy as np
import pandas as pd
import sys
import mykmeanssp

if len(sys.argv) not in [5, 6]:
    print("An Error Has Occurred")
    sys.exit()

try:
    k = sys.argv[1]

    if len(sys.argv) == 5:
        max_iter = 300
        eps = sys.argv[2]
        file_1 = sys.argv[3]
        file_2 = sys.argv[4]
    else:
        max_iter = sys.argv[2]
        eps = sys.argv[3]
        file_1 = sys.argv[4]
        file_2 = sys.argv[5]

    f1 = file_1[-4:]
    f2 = file_2[-4:]
    if (f1 not in [".txt", ".csv"]) or (f2 not in [".txt", ".csv"]):
        print("NA")
        sys.exit()

    if max_iter.isdigit() == False or not (1 < int(max_iter) < 1000) or max_iter[0] == "0":
        print("Invalid maximum iteration!")
        sys.exit()

    if float(eps) < 0:
        print("Invalid epsilon!")
        sys.exit()

except:
    print("An Error Has Occurred")
    sys.exit()

df1 = pd.read_csv(file_1, header=None)
df2 = pd.read_csv(file_2, header=None)

key_column = 0
df1[key_column] = df1[key_column].astype(int)
df2[key_column] = df2[key_column].astype(int)

df = pd.merge(df1, df2, on=key_column, how='inner').sort_values(by=key_column)
original_keys = df[key_column].to_list()
points = df.iloc[:, 1:].to_numpy()

np.random.seed(1234)
initial_centroids_idx = [np.random.choice(range(points.shape[0]))]
for _ in range(1, int(k)):
    distances = np.min([np.linalg.norm(points - points[c], axis=1) ** 2 for c in initial_centroids_idx], axis=0)
    prob = distances / distances.sum()
    new_centroid = np.random.choice(range(points.shape[0]), p=prob)
    initial_centroids_idx.append(new_centroid)

initial_centroids_keys = [original_keys[idx] for idx in initial_centroids_idx]
initial_centroids = points[initial_centroids_idx]

final_centroids = mykmeanssp.fit(initial_centroids.tolist(), points.tolist(), int(k), int(max_iter), float(eps))

print(','.join(map(str, initial_centroids_keys)))
for centroid in final_centroids:
    print(','.join(f"{coord:.4f}" for coord in centroid))
