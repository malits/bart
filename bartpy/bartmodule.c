#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>
// TODO: For the sake of organization, move these elsewhere
// E.g., separate python files for separate libraries.
// Ultimately automate from source code in C.

typedef struct {
    PyObject_HEAD;
    float version;
} BartObject;

static PyObject * Bart_new(PyTypeObject * type, PyObject *args, PyObject *kwds) 
{
    BartObject *self;
    self = (BartObject *) type->tp_alloc(type, 0);
    return (PyObject *) self;
}

static int Bart_init(BartObject * self, PyObject * args, PyObject * kwds)
{
    self->version = 1.0;
    return 0;
}

static PyObject * Bart_help(BartObject *self, PyObject * Py_UNUSED(ignored))
{
    return PyUnicode_FromString("BART: Toolbox for Computational Magnetic Resonance Imaging (MRI)");
}

static PyObject * Bart_phantom(BartObject * self, PyObject * args) 
{
    return PyUnicode_FromString("Shepp-Logan Phantom");
}

static PyMemberDef Bart_members[] = {
    {"version", T_FLOAT, offsetof(BartObject, version), READONLY, "version"},
    {NULL}
};

static PyMethodDef Bart_methods[] = {
    {"help", (PyCFunction) Bart_help, METH_NOARGS, "Help string."},
    {"phantom", (PyCFunction) Bart_phantom, METH_VARARGS, "Simple Numerical Phantom"},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject Bart = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "bart.Bart",
    .tp_doc = "BART",
    .tp_basicsize = sizeof(BartObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = Bart_new,
    .tp_init = (initproc) Bart_init,
    .tp_methods = Bart_methods,
    .tp_members = Bart_members,
};

static PyModuleDef bartmodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "bart",
    .m_doc = "BART: Toolbox for Computational Magnetic Resonance Imaging (MRI)",
    .m_size = -1,
};

PyMODINIT_FUNC
PyInit_bart(void)
{
    PyObject *m;
    if (PyType_Ready(&Bart) < 0)
        return NULL;

    m = PyModule_Create(&bartmodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&Bart);
    if (PyModule_AddObject(m, "Bart", (PyObject *) &Bart) < 0) {
        Py_DECREF(&Bart);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
