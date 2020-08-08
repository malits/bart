#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

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

// TODO: This was breaking class methods
// static PyObject * bart_getattr(BartObject * self, char * name)
// {
//     if (strcmp(name, "version") == 0)
//         return PyFloat_FromDouble(self->version);

//     PyErr_Format(PyExc_AttributeError,
//                  "No attribute '%.400s'",
//                  name);
//     return NULL;
// }

// static int bart_setattr(BartObject *obj, char *name, PyObject *v)
// {
//     PyErr_Format(PyExc_RuntimeError, "Read-only attribute: %s", name);
//     return -1;
// }

static PyObject * Bart_help(BartObject *self, PyObject * Py_UNUSED(ignored))
{
    return PyUnicode_FromString("BART: Toolbox for Computational Magnetic Resonance Imaging (MRI)");
}

static PyMemberDef Bart_members[] = {
    {"version", T_OBJECT_EX, offsetof(BartObject, version), 0, "version"},
    {NULL}
};

static PyMethodDef Bart_methods[] = {
    {"help", (PyCFunction) Bart_help, METH_NOARGS, "Help string."},
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
    //.tp_getattr = (getattrfunc) bart_getattr,
    //.tp_setattr = (setattrfunc) bart_setattr,
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