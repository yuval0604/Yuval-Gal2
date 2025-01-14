#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

static PyObject* fit(PyObject* self, PyObject* args) {
    PyObject *centroids_obj, *points_obj;
    int k, max_iter;
    double eps;

    if (!PyArg_ParseTuple(args, "OOiid", &centroids_obj, &points_obj, &k, &max_iter, &eps)) {
        return NULL;
    }

    // Parse Python list to C arrays
    int num_points = (int)PyList_Size(points_obj);
    int dim = (int)PyList_Size(PyList_GetItem(points_obj, 0));
    double **centroids = (double **)malloc(k * sizeof(double *));
    double **points = (double **)malloc(num_points * sizeof(double *));

    for (int i = 0; i < k; i++) {
        centroids[i] = (double *)malloc(dim * sizeof(double));
        for (int j = 0; j < dim; j++) {
            centroids[i][j] = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(centroids_obj, i), j));
        }
    }

    for (int i = 0; i < num_points; i++) {
        points[i] = (double *)malloc(dim * sizeof(double));
        for (int j = 0; j < dim; j++) {
            points[i][j] = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(points_obj, i), j));
        }
    }

    // K-means main loop
    for (int iter = 0; iter < max_iter; iter++) {
        // Assign points to nearest centroid and update centroids
        int changed = 0;
        double **new_centroids = calloc(k, sizeof(double *));
        int *counts = calloc(k, sizeof(int));

        for (int i = 0; i < num_points; i++) {
            int closest = 0;
            double min_dist = INFINITY;

            for (int j = 0; j < k; j++) {
                double dist = 0.0;
                for (int d = 0; d < dim; d++) {
                    double diff = points[i][d] - centroids[j][d];
                    dist += diff * diff;
                }
                if (dist < min_dist) {
                    min_dist = dist;
                    closest = j;
                }
            }
            counts[closest]++;
            for (int d = 0; d < dim; d++) {
                if (new_centroids[closest] == NULL) {
                    new_centroids[closest] = calloc(dim, sizeof(double));
                }
                new_centroids[closest][d] += points[i][d];
            }
        }

        // Update centroids
        for (int j = 0; j < k; j++) {
            for (int d = 0; d < dim; d++) {
                new_centroids[j][d] /= counts[j] ? counts[j] : 1;
                if (fabs(new_centroids[j][d] - centroids[j][d]) >= eps) {
                    changed = 1;
                }
                centroids[j][d] = new_centroids[j][d];
            }
            free(new_centroids[j]);
        }

        free(counts);
        free(new_centroids);

        if (!changed) break;
    }

    // Convert centroids back to Python list
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
    for (int i = 0; i < num_points; i++) {
        free(points[i]);
    }
    free(points);

    return result;
}

static PyMethodDef MyKMeansMethods[] = {
    {"fit", fit, METH_VARARGS, "K-means clustering implementation"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef mykmeansspmodule = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",
    NULL,
    -1,
    MyKMeansMethods
};

PyMODINIT_FUNC PyInit_mykmeanssp(void) {
    return PyModule_Create(&mykmeansspmodule);
}
