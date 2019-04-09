#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "Python.h"
#include "numpy/arrayobject.h"

#include "fortran.h"
#include "cube.c"


static PyObject * pytest_dotest(PyObject * self, PyObject * args) {
    PyArrayObject * data, * var;

    if (!PyArg_ParseTuple(args, "O&O&",
                          PyArray_Converter, &data,
                          PyArray_Converter, &var))
        return NULL;

    PyObject_Print((PyObject *) data, stdout, 0);
    printf("\n");

    int ndim = PyArray_NDIM(data);
    npy_intp * dims = PyArray_DIMS(data);

    printf("%d: ", ndim);
    for (int i=0; i<ndim; ++i)
        printf("%ld,", dims[i]);
    printf("\n");

    void * cube_ptr = cube_init(dims, PyArray_DATA(data), PyArray_DATA(var));
    cube_flats(cube_ptr);
    cube_print(cube_ptr);

    void * white_ptr, * red_ptr, * green_ptr, * blue_ptr, * inv_var_ptr;
    cube_get_flats(cube_ptr, &white_ptr, &red_ptr, &green_ptr, &blue_ptr, &inv_var_ptr);

    printf("And the white is:\n");
    PyArrayObject * white = (PyArrayObject *) PyArray_SimpleNewFromData(2, &dims[1], NPY_FLOAT32, white_ptr);
    PyObject_Print((PyObject *) white, stdout, 0);
    printf("\n");

    Py_DECREF(white);


    double lw[] = {0.26, 0.7, 1, 0.7, 0.26};
//    muselet_step1(cube_ptr, 3, 20, 5, lw, callback);


    Py_DECREF(data);
    Py_DECREF(var);
    Py_RETURN_NONE;
}

//static char * dir = "/users/kosio/desktop/nb/";
static void muselet_nb_callback(void * c, float * imslice_ptr, intptr_t * i) {
    cube * ccube = (cube *) c;
    char * fname = malloc(strlen(ccube->nbdir)+14);
    strcpy(fname, ccube->nbdir);

    char * name = malloc(12);
    sprintf(name, "nb%04ld.fits", (*i));
    strcat(fname, name);

    cube_write_img(ccube, fname, imslice_ptr);
    puts(fname);

    free(fname);
    free(name);
}

static PyObject * pylet_step1(PyObject * self, PyObject * args) {
    cube c;
    Py_ssize_t left, cw;
    PyArrayObject * lw;

    const char *white_name, *R_name, *G_name, *B_name, *inv_var_name;

    if (!PyArg_ParseTuple(args, "(ss)s(sssss)nnO&",
                          &c.dataname, &c.varname, &c.nbdir,
                          &white_name, &R_name, &G_name, &B_name, &inv_var_name,
                          &left, &cw,
                          PyArray_Converter, &lw))
        return NULL;

    cube_load(&c);
    cube_flats(c.fcube);

    void *white_ptr, *R_ptr, *G_ptr, *B_ptr, *inv_var_ptr;
    cube_get_flats(c.fcube, &white_ptr, &R_ptr, &G_ptr, &B_ptr, &inv_var_ptr);

    cube_write_img(&c, white_name, white_ptr);
    cube_write_img(&c, R_name, R_ptr);
    cube_write_img(&c, G_name, G_ptr);
    cube_write_img(&c, B_name, B_ptr);
    cube_write_img(&c, inv_var_name, inv_var_ptr);

    muselet_step1(c.fcube, left, cw, PyArray_SIZE(lw), PyArray_DATA(lw), muselet_nb_callback, &c);

    cube_deallocate(c.fcube);
    cube_free(&c);

    Py_RETURN_NONE;
}


static PyMethodDef incubator_methods[] = {
        {"dotest", pytest_dotest, METH_VARARGS, "Do a test."},
        {"_pylet_step1", pylet_step1, METH_VARARGS, "Muselet. Step 1."},
        {NULL, NULL, 0, NULL}   // Sentinel
};
static PyModuleDef incubator = {
        PyModuleDef_HEAD_INIT,
        "incubator",        // name of module
        "Incubator by Kosio Karchev. Looks into cubes.",        // module documentation
        -1,                 // size of per-interpreter state of the module, or -1 if the module keeps state in global variables
        incubator_methods   // PyMethodDef instance
};
PyMODINIT_FUNC PyInit_incubator(void) {
    import_array();
    return PyModule_Create(&incubator);
}