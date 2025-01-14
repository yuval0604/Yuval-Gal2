import numpy as np
import pandas as pd
import sys
import mykmeanssp

def main():
    if len(sys.argv) < 5:
        print("An Error Has Occurred")
        return

    try:
        k = int(sys.argv[1])
        max_iter = int(sys.argv[2]) if len(sys.argv) > 5 else 300
        eps = float(sys.argv[3])
        file_1 = sys.argv[4]
        file_2 = sys.argv[5]
    except ValueError:
        print("An Error Has Occurred")
        return

    df1 = pd.read_csv(file_1)
    df2 = pd.read_csv(file_2)

    df = pd.merge(df1, df2, on=df1.columns[0]).sort_values(by=df1.columns[0])
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

if __name__ == "__main__":
    main()
