/*********************************************************
Model Name      : nothing
Filename        : suffix_nothing_again.mod
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
    inline static double forty_three_nothing();


    /** connect global (scalar) variables to hoc -- */
    static DoubScal hoc_scalar_double[] = {
        {nullptr, nullptr}
    };


    /** connect global (array) variables to hoc -- */
    static DoubVec hoc_vector_double[] = {
        {nullptr, nullptr, 0}
    };


    /* declaration of user functions */
    static void _hoc_forty_three();
    static double _npy_forty_three(Prop* _prop);


    /* connect user functions to hoc names */
    static VoidFunc hoc_intfunc[] = {
        {"forty_three", _hoc_forty_three},
        {nullptr, nullptr}
    };
    static NPyDirectMechFunc npy_direct_func_proc[] = {
        {"forty_three", _npy_forty_three},
        {nullptr, nullptr}
    };
    static void _hoc_forty_three() {
        double _r = 0.0;
        _r = forty_three_nothing();
        hoc_retpushx(_r);
    }
    static double _npy_forty_three(Prop* _prop) {
        double _r = 0.0;
        _r = forty_three_nothing();
        return(_r);
    }


    inline double forty_three_nothing() {
        double ret_forty_three = 0.0;
        ret_forty_three = 43.0;
        return ret_forty_three;
    }


    extern "C" void _suffix_nothing_again_reg() {
        hoc_register_var(hoc_scalar_double, hoc_vector_double, hoc_intfunc);
    }
}
