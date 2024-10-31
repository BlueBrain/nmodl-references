/*********************************************************
Model Name      : nothing
Filename        : nothing.mod
NMODL Version   : 7.7.0
Vectorized      : true
Threadsafe      : true
Created         : DATE
Simulator       : NEURON
Backend         : C++ (api-compatibility)
NMODL Compiler  : VERSION
*********************************************************/

#include <Eigen/Dense>
#include <Eigen/LU>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "mech_api.h"
#include "neuron/cache/mechanism_range.hpp"
#include "nrniv_mf.h"
#include "section_fwd.hpp"


namespace neuron {


    /* Mechanism procedures and functions */
    inline static double get_foo_nothing();
}


using namespace neuron;


// Setup for VERBATIM
// Begin VERBATIM

double foo = 42.0;
// End VERBATIM
// End of cleanup for VERBATIM


namespace neuron {


    /** connect global (scalar) variables to hoc -- */
    static DoubScal hoc_scalar_double[] = {
        {nullptr, nullptr}
    };


    /** connect global (array) variables to hoc -- */
    static DoubVec hoc_vector_double[] = {
        {nullptr, nullptr, 0}
    };


    /* declaration of user functions */
    static void _hoc_get_foo();
    static double _npy_get_foo(Prop* _prop);


    /* connect user functions to hoc names */
    static VoidFunc hoc_intfunc[] = {
        {"get_foo", _hoc_get_foo},
        {nullptr, nullptr}
    };
    static NPyDirectMechFunc npy_direct_func_proc[] = {
        {"get_foo", _npy_get_foo},
        {nullptr, nullptr}
    };
    static void _hoc_get_foo() {
        double _r = 0.0;
        _r = get_foo_nothing();
        hoc_retpushx(_r);
    }
    static double _npy_get_foo(Prop* _prop) {
        double _r = 0.0;
        _r = get_foo_nothing();
        return(_r);
    }


    inline double get_foo_nothing() {
        double ret_get_foo = 0.0;
        // Setup for VERBATIM
        #define t nt->_t
        // Begin VERBATIM
        
            return foo;
        // End VERBATIM
        #undef t
        // End of cleanup for VERBATIM

        return ret_get_foo;
    }


    extern "C" void _nothing_reg() {
        hoc_register_var(hoc_scalar_double, hoc_vector_double, hoc_intfunc);
    }
}
