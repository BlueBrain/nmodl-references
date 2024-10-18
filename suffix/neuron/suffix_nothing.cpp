/*********************************************************
Model Name      : nothing
Filename        : suffix_nothing.mod
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
    inline static double forty_two_plus_x_nothing(double _lx);
    inline static double twice_forty_two_plus_x_nothing(double _lx);


    /** connect global (scalar) variables to hoc -- */
    static DoubScal hoc_scalar_double[] = {
        {nullptr, nullptr}
    };


    /** connect global (array) variables to hoc -- */
    static DoubVec hoc_vector_double[] = {
        {nullptr, nullptr, 0}
    };


    /* declaration of user functions */
    static void _hoc_forty_two_plus_x();
    static double _npy_forty_two_plus_x(Prop* _prop);
    static void _hoc_twice_forty_two_plus_x();
    static double _npy_twice_forty_two_plus_x(Prop* _prop);


    /* connect user functions to hoc names */
    static VoidFunc hoc_intfunc[] = {
        {"forty_two_plus_x", _hoc_forty_two_plus_x},
        {"twice_forty_two_plus_x", _hoc_twice_forty_two_plus_x},
        {nullptr, nullptr}
    };
    static NPyDirectMechFunc npy_direct_func_proc[] = {
        {"forty_two_plus_x", _npy_forty_two_plus_x},
        {"twice_forty_two_plus_x", _npy_twice_forty_two_plus_x},
        {nullptr, nullptr}
    };
    static void _hoc_forty_two_plus_x() {
        double _r = 0.0;
        _r = forty_two_plus_x_nothing(*getarg(1));
        hoc_retpushx(_r);
    }
    static double _npy_forty_two_plus_x(Prop* _prop) {
        double _r = 0.0;
        _r = forty_two_plus_x_nothing(*getarg(1));
        return(_r);
    }
    static void _hoc_twice_forty_two_plus_x() {
        double _r = 0.0;
        _r = twice_forty_two_plus_x_nothing(*getarg(1));
        hoc_retpushx(_r);
    }
    static double _npy_twice_forty_two_plus_x(Prop* _prop) {
        double _r = 0.0;
        _r = twice_forty_two_plus_x_nothing(*getarg(1));
        return(_r);
    }


    inline double forty_two_plus_x_nothing(double _lx) {
        double ret_forty_two_plus_x = 0.0;
        ret_forty_two_plus_x = 42.0 + _lx;
        return ret_forty_two_plus_x;
    }


    inline double twice_forty_two_plus_x_nothing(double _lx) {
        double ret_twice_forty_two_plus_x = 0.0;
        ret_twice_forty_two_plus_x = 2.0 * forty_two_plus_x_nothing(_lx);
        return ret_twice_forty_two_plus_x;
    }


    extern "C" void _suffix_nothing_reg() {
        hoc_register_var(hoc_scalar_double, hoc_vector_double, hoc_intfunc);
    }
}
