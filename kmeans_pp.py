import numpy as np
import pandas as pd
import sys
import mykmeanssp

if len(sys.argv) not in [5, 6]:
    print("An Error Has Occurred")
    sys.exit()

try:
    k = int(sys.argv[1])
    
    if len(sys.argv) == 5:
        max_iter = 300
        eps = float(sys.argv[2])
        file_1 = sys.argv[3]
        file_2 = sys.argv[4]
    else:
        max_iter = int(sys.argv[2])
        eps = float(sys.argv[3])
        file_1 = sys.argv[4]
        file_2 = sys.argv[5]

    
    f1 = file_1[-4:]
    f2 = file_2[-4:]
    if (f1 not in [".txt", ".csv"]) or (f2 not in [".txt", ".csv"]):
        print("NA")
        sys.exit()
    
    if not (1 < max_iter < 1000):
        print("Invalid maximum iteration!")
        sys.exit()
    
    if eps < 0:
        print("Invalid epsilon!")
        sys.exit()

except ValueError:
    print("An Error Has Occurred")
    sys.exit()


df1 = pd.read_csv(file_1, header=None)
df2 = pd.read_csv(file_2, header=None)


key_column = 0  
df1[key_column] = df1[key_column].astype(int)
df2[key_column] = df2[key_column].astype(int)


df = pd.merge(df1, df2, on=key_column, how='inner').sort_values(by=key_column)

if df.empty:
    print("An Error Has Occurred")
    sys.exit()


points = df.iloc[:, 1:].values


np.random.seed(1234)
initial_centroids_idx = [np.random.choice(range(points.shape[0]))]
for _ in range(1, k):
    distances = np.min([np.linalg.norm(points - points[c], axis=1) ** 2 for c in initial_centroids_idx], axis=0)
    prob = distances / distances.sum()
    new_centroid = np.random.choice(range(points.shape[0]), p=prob)
    initial_centroids_idx.append(new_centroid)


initial_centroids = points[initial_centroids_idx]
final_centroids = mykmeanssp.fit(initial_centroids.tolist(), points.tolist(), k, max_iter, eps)


print(','.join(map(str, initial_centroids_idx)))
for centroid in final_centroids:
    print(','.join(f"{coord:.4f}" for coord in centroid))
