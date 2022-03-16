#include <Python.h>
#include <stdbool.h>
#if __APPLE__
#include <sys/types.h>
#endif
#include <sys/ptrace.h>

#define UNUSED(arg) arg __attribute__((unused))

char python_ptrace_DOCSTR[] =
"ptrace(command: int, pid: int, arg1=0, arg2=0, check_errno=False): call ptrace syscall.\r\n"
"Raise a ValueError on error.\r\n"
"Returns an unsigned integer.\r\n";

static PyObject *PyExc_PtraceError;

static bool cpython_cptrace(
    unsigned int request,
    pid_t pid,
    void *arg1,
    void *arg2,
    bool check_errno,
    unsigned long *result)
{
    unsigned long ret;
    errno = 0;
    ret = ptrace(request, pid, arg1, arg2);
    if ((long)ret == -1) {
        /**
         * peek operations may returns -1 with errno=0: it's not an error.
         * For other operations, -1 is always an error
         */
        if (!check_errno || errno) {
            PyObject *message = PyUnicode_FromFormat(
                "ptrace(request=%u, pid=%i, %p, %p) "
                "error #%i: %s",
                request, pid, arg1, arg2,
                errno, strerror(errno));
            PyObject *argList = Py_BuildValue("Oii", message, errno, pid);
            PyObject *p_exc = PyObject_CallObject(PyExc_PtraceError, argList);
            PyErr_SetObject(PyExc_PtraceError, p_exc);
            Py_DECREF(message);
            Py_DECREF(argList);
            return false;
        }
    }
    if (result)
        *result = ret;
    return true;
}

static PyObject* cpython_ptrace(PyObject* UNUSED(self), PyObject *args, PyObject *keywds)
{
    unsigned long result;
    unsigned int request;
    pid_t pid;
    unsigned long long arg1 = 0;
    unsigned long long arg2 = 0;
    bool check_errno = false;
    PyObject* check_errno_p = NULL;
    static char *kwlist[] = {"request", "pid", "arg1", "arg2", "check_errno", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds,
        "Ii|KKO", kwlist,
        &request, &pid, &arg1, &arg2, &check_errno_p
    ))
    {
        return NULL;
    }

    if (check_errno_p) {
        check_errno = PyObject_IsTrue(check_errno_p);
    }

    if (cpython_cptrace(request, pid, (void*)arg1, (void*)arg2, check_errno, &result))
        return PyLong_FromUnsignedLong(result);
    else
        return NULL;
}

static PyMethodDef module_methods[] = {
    {"ptrace", (PyCFunction)cpython_ptrace, METH_VARARGS | METH_KEYWORDS, python_ptrace_DOCSTR},
    {NULL, NULL, 0, NULL}
};

PyDoc_STRVAR(module_doc,
"ptrace module written in C");

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "cptrace",
    module_doc,
    0,
    module_methods,
    NULL
};
#endif

PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
PyInit_cptrace(void)
#else
initcptrace(void)
#endif
{
    PyObject *cls_name = PyUnicode_FromString("ptrace.errors");
    PyObject *ptrace_errors = PyImport_Import(cls_name);
    PyObject *exc_name = PyUnicode_FromString("PtraceError");
    PyExc_PtraceError = PyObject_GetAttr(ptrace_errors, exc_name);

    Py_DECREF(cls_name);
    Py_DECREF(ptrace_errors);
    Py_DECREF(exc_name);

    if (PyExc_PtraceError == NULL) {
        return NULL;
    }

#if PY_MAJOR_VERSION >= 3
    return PyModule_Create(&module_def);
#else
    (void)Py_InitModule3("cptrace", module_methods, module_doc);
#endif
}

