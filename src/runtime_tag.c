/* Marker extension: forces a platform-specific wheel tag. No runtime API. */
#include <Python.h>

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_runtime_tag", NULL, -1, NULL
};

PyMODINIT_FUNC PyInit__runtime_tag(void) {
    return PyModule_Create(&module);
}
