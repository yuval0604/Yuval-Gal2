#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

// Function to calculate squared Euclidean distance between two points
double euclidean_distance(double *point1, double *point2, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim; i++) {
        double diff = point1[i] - point2[i];
        sum += diff * diff;
    }
    return sum;  // Return squared distance
}

// Function to assign points to the nearest centroid
void assign_points_to_clusters(double **points, double **centroids, int *cluster_assignments, int num_points, int k, int dim) {
    for (int i = 0; i < num_points; i++) {
        int closest_centroid = 0;
        double min_distance = euclidean_distance(points[i], centroids[0], dim);
        
        for (int j = 1; j < k; j++) {
            double dist = euclidean_distance(points[i], centroids[j], dim);
            if (dist < min_distance) {
                min_distance = dist;
                closest_centroid = j;
            }
        }
        cluster_assignments[i] = closest_centroid;  // Assign the point to the closest centroid
    }
}

// Function to update centroids based on the assigned points
void update_centroids(double **points, double **centroids, int *cluster_assignments, int num_points, int k, int dim) {
    int *counts = (int *)calloc(k, sizeof(int));  // Store the number of points per cluster
    double **sums = (double **)malloc(k * sizeof(double *));
    
    for (int i = 0; i < k; i++) {
        sums[i] = (double *)calloc(dim, sizeof(double));
    }

    // Sum points in each cluster
    for (int i = 0; i < num_points; i++) {
        int cluster = cluster_assignments[i];
        counts[cluster]++;
        for (int j = 0; j < dim; j++) {
            sums[cluster][j] += points[i][j];
        }
    }

    // Calculate the new centroids (mean of points in each cluster)
    for (int i = 0; i < k; i++) {
        if (counts[i] > 0) {
            for (int j = 0; j < dim; j++) {
                centroids[i][j] = sums[i][j] / counts[i];
            }
        }
    }

    // Free memory
    for (int i = 0; i < k; i++) {
        free(sums[i]);
    }
    free(sums);
    free(counts);
}

// Main K-means function
static PyObject* fit(PyObject* self, PyObject* args) {
    PyObject *centroids_obj, *points_obj;
    int k, max_iter;
    double eps;

    // Parse Python arguments: initial centroids, data points, k, max_iter, eps
    if (!PyArg_ParseTuple(args, "OOiid", &centroids_obj, &points_obj, &k, &max_iter, &eps)) {
        return NULL;  // Return NULL if parsing failed
    }

    int num_points = PyList_Size(points_obj);
    int dim = PyList_Size(PyList_GetItem(points_obj, 0));
    double **centroids = malloc(k * sizeof(double *));
    double **points = malloc(num_points * sizeof(double *));
    int *cluster_assignments = malloc(num_points * sizeof(int));

    // Convert Python lists to C arrays for centroids and points
    for (int i = 0; i < k; i++) {
        centroids[i] = malloc(dim * sizeof(double));
        for (int j = 0; j < dim; j++) {
            centroids[i][j] = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(centroids_obj, i), j));
        }
    }

    for (int i = 0; i < num_points; i++) {
        points[i] = malloc(dim * sizeof(double));
        for (int j = 0; j < dim; j++) {
            points[i][j] = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(points_obj, i), j));
        }
    }

    // Main K-means loop
    for (int iter = 0; iter < max_iter; iter++) {
        double prev_centroids[k][dim];  // Store previous centroids for comparison
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < dim; j++) {
                prev_centroids[i][j] = centroids[i][j];
            }
        }

        assign_points_to_clusters(points, centroids, cluster_assignments, num_points, k, dim);
        update_centroids(points, centroids, cluster_assignments, num_points, k, dim);

        // Check convergence (if change in centroids is below eps for all centroids)
        int converged = 1;
        for (int i = 0; i < k; i++) {
            if (euclidean_distance(prev_centroids[i], centroids[i], dim) >= eps) {
                converged = 0;
                break;
            }
        }

        if (converged) break;  // Exit if converged
    }

    // Convert final centroids to Python list
    PyObject *result = PyList_New(k);
    for (int i = 0; i < k; i++) {
        PyObject *centroid = PyList_New(dim);
        for (int j = 0; j < dim; j++) {
            PyList_SetItem(centroid, j, PyFloat_FromDouble(centroids[i][j]));
        }
        PyList_SetItem(result, i, centroid);
        free(centroids[i]);
    }
    free(centroids);
    free(cluster_assignments);
    for (int i = 0; i < num_points; i++) {
        free(points[i]);
    }
    free(points);

    return result;  // Return final centroids to Python
}

// Define the methods of the module
static PyMethodDef MyKMeansMethods[] = {
    {"fit", fit, METH_VARARGS, "K-means clustering implementation"},
    {NULL, NULL, 0, NULL}
};

// Define the module
static struct PyModuleDef mykmeansspmodule = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",
    NULL,
    -1,
    MyKMeansMethods
};

// Initialize the module
PyMODINIT_FUNC PyInit_mykmeanssp(void) {
    return PyModule_Create(&mykmeansspmodule);
}
