%module phantom //module name
%{ // this is copied verbatim to the file; put imports and defs here
    #define SWIG_FILE_WITH_INIT

    #include <math.h>
    #include <complex.h>
    #include <string.h>
    #include <stdbool.h>
    #include <assert.h>

    #include <lapacke.h>

    #include "../src/num/multind.h"
    #include "../src/num/loop.h"
    #include "../src/num/flpmath.h"
    #include "../src/num/splines.h"

    #include "../src/misc/misc.h"
    #include "../src/misc/mri.h"
    #include "../src/misc/debug.h"

    #include "../src/geom/logo.h"

    #include "../src/simu/sens.h"
    #include "../src/simu/coil.h"
    #include "../src/simu/shape.h"
    #include "../src/simu/shepplogan.h"

    #include "../src/simu/phantom.h"

    #include "../src/num/vecops.h"

    #define MAX_COILS 8
    #define COIL_COEFF 5

%} 

%define _Complex complex %enddef //TODO: find a cleaner way to handle this
%define` _Bool bool %enddef
%include "../src/simu/phantom.h"