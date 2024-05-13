/*********************************************************
Model Name      : test_func_proc
Filename        : func_proc.mod
NMODL Version   : 7.7.0
Vectorized      : true
Threadsafe      : true
Created         : Mon May 13 13:22:42 2024
Simulator       : NEURON
Backend         : C++ (api-compatibility)
NMODL Compiler  : 0.0 [43dfc32 2024-05-13 13:21:03 +0000]
*********************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "mech_api.h"
#include "neuron/cache/mechanism_range.hpp"
#include "nrniv_mf.h"
#include "section_fwd.hpp"

/* NEURON global macro definitions */
/* VECTORIZED */
#define NRN_VECTORIZED 1

static constexpr auto number_of_datum_variables = 0;
static constexpr auto number_of_floating_point_variables = 2;

namespace {
template <typename T>
using _nrn_mechanism_std_vector = std::vector<T>;
using _nrn_model_sorted_token = neuron::model_sorted_token;
using _nrn_mechanism_cache_range = neuron::cache::MechanismRange<number_of_floating_point_variables, number_of_datum_variables>;
using _nrn_mechanism_cache_instance = neuron::cache::MechanismInstance<number_of_floating_point_variables, number_of_datum_variables>;
using _nrn_non_owning_id_without_container = neuron::container::non_owning_identifier_without_container;
template <typename T>
using _nrn_mechanism_field = neuron::mechanism::field<T>;
template <typename... Args>
void _nrn_mechanism_register_data_fields(Args&&... args) {
    neuron::mechanism::register_data_fields(std::forward<Args>(args)...);
}
}  // namespace

Prop* hoc_getdata_range(int type);


namespace neuron {
    #ifndef NRN_PRCELLSTATE
    #define NRN_PRCELLSTATE 0
    #endif


    /** channel information */
    static const char *mechanism_info[] = {
        "7.7.0",
        "test_func_proc",
        0,
        "x_test_func_proc",
        0,
        0,
        0
    };


    /* NEURON global variables */
    static int mech_type;
    static Prop* _extcall_prop;
    /* _prop_id kind of shadows _extcall_prop to allow validity checking. */
    static _nrn_non_owning_id_without_container _prop_id{};
    static int hoc_nrnpointerindex = -1;
    static _nrn_mechanism_std_vector<Datum> _extcall_thread;


    /** all global variables */
    struct test_func_proc_Store {
    };
    static_assert(std::is_trivially_copy_constructible_v<test_func_proc_Store>);
    static_assert(std::is_trivially_move_constructible_v<test_func_proc_Store>);
    static_assert(std::is_trivially_copy_assignable_v<test_func_proc_Store>);
    static_assert(std::is_trivially_move_assignable_v<test_func_proc_Store>);
    static_assert(std::is_trivially_destructible_v<test_func_proc_Store>);
    test_func_proc_Store test_func_proc_global;


    /** all mechanism instance variables and global variables */
    struct test_func_proc_Instance  {
        double* x{};
        double* v_unused{};
        test_func_proc_Store* global{&test_func_proc_global};
    };


    struct test_func_proc_NodeData  {
        int const * nodeindices;
        double const * node_voltages;
        double * node_diagonal;
        double * node_rhs;
        int nodecount;
    };


    static test_func_proc_Instance make_instance_test_func_proc(_nrn_mechanism_cache_range& _ml) {
        return test_func_proc_Instance {
            _ml.template fpfield_ptr<0>(),
            _ml.template fpfield_ptr<1>()
        };
    }


    static test_func_proc_NodeData make_node_data_test_func_proc(NrnThread& _nt, Memb_list& _ml_arg) {
        return test_func_proc_NodeData {
            _ml_arg.nodeindices,
            _nt.node_voltage_storage(),
            _nt.node_d_storage(),
            _nt.node_rhs_storage(),
            _ml_arg.nodecount
        };
    }


    static void nrn_alloc_test_func_proc(Prop* _prop) {
        Prop *prop_ion{};
        Datum *_ppvar{};
        _nrn_mechanism_cache_instance _ml_real{_prop};
        auto* const _ml = &_ml_real;
        size_t const _iml{};
        assert(_nrn_mechanism_get_num_vars(_prop) == 2);
        /*initialize range parameters*/
    }


    /* Neuron setdata functions */
    extern void _nrn_setdata_reg(int, void(*)(Prop*));
    static void _setdata(Prop* _prop) {
        _extcall_prop = _prop;
        _prop_id = _nrn_get_prop_id(_prop);
    }
    static void _hoc_setdata() {
        Prop *_prop = hoc_getdata_range(mech_type);
        _setdata(_prop);
        hoc_retpushx(1.);
    }
    /* Mechanism procedures and functions */
    inline double x_plus_a_test_func_proc(_nrn_mechanism_cache_range* _ml, test_func_proc_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double a);
    inline int set_x_42_test_func_proc(_nrn_mechanism_cache_range* _ml, test_func_proc_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* _nt);
    inline int set_x_a_test_func_proc(_nrn_mechanism_cache_range* _ml, test_func_proc_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double a);
    inline int set_a_x_test_func_proc(_nrn_mechanism_cache_range* _ml, test_func_proc_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* _nt);


    /** connect global (scalar) variables to hoc -- */
    static DoubScal hoc_scalar_double[] = {
        {nullptr, nullptr}
    };


    /** connect global (array) variables to hoc -- */
    static DoubVec hoc_vector_double[] = {
        {nullptr, nullptr, 0}
    };


    /* declaration of user functions */
    static void _hoc_set_x_42(void);
    static void _hoc_set_x_a(void);
    static void _hoc_set_a_x(void);
    static void _hoc_x_plus_a(void);
    static double _npy_set_x_42(Prop*);
    static double _npy_set_x_a(Prop*);
    static double _npy_set_a_x(Prop*);
    static double _npy_x_plus_a(Prop*);


    /* connect user functions to hoc names */
    static VoidFunc hoc_intfunc[] = {
        {"setdata_test_func_proc", _hoc_setdata},
        {"set_x_42_test_func_proc", _hoc_set_x_42},
        {"set_x_a_test_func_proc", _hoc_set_x_a},
        {"set_a_x_test_func_proc", _hoc_set_a_x},
        {"x_plus_a_test_func_proc", _hoc_x_plus_a},
        {0, 0}
    };
    static NPyDirectMechFunc npy_direct_func_proc[] = {
        {"set_x_42", _npy_set_x_42},
        {"set_x_a", _npy_set_x_a},
        {"set_a_x", _npy_set_a_x},
        {"x_plus_a", _npy_x_plus_a},
    };
    static void _hoc_set_x_42(void) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* _nt;
        if (!_prop_id) {
            hoc_execerror("No data for set_x_42_test_func_proc. Requires prior call to setdata_test_func_proc and that the specified mechanism instance still be in existence.", NULL);
        }
        Prop* _local_prop = _extcall_prop;
        _nrn_mechanism_cache_instance _ml_real{_local_prop};
        auto* const _ml = &_ml_real;
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        _nt = nrn_threads;
        auto inst = make_instance_test_func_proc(_ml_real);
        _r = 1.;
        set_x_42_test_func_proc(_ml, inst, id, _ppvar, _thread, _nt);
        hoc_retpushx(_r);
    }
    static double _npy_set_x_42(Prop* _prop) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* _nt;
        _nrn_mechanism_cache_instance _ml_real{_prop};
        auto* const _ml = &_ml_real;
        size_t const id{};
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        _nt = nrn_threads;
        auto inst = make_instance_test_func_proc(_ml_real);
        _r = 1.;
        set_x_42_test_func_proc(_ml, inst, id, _ppvar, _thread, _nt);
        return(_r);
    }
    static void _hoc_set_x_a(void) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* _nt;
        if (!_prop_id) {
            hoc_execerror("No data for set_x_a_test_func_proc. Requires prior call to setdata_test_func_proc and that the specified mechanism instance still be in existence.", NULL);
        }
        Prop* _local_prop = _extcall_prop;
        _nrn_mechanism_cache_instance _ml_real{_local_prop};
        auto* const _ml = &_ml_real;
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        _nt = nrn_threads;
        auto inst = make_instance_test_func_proc(_ml_real);
        _r = 1.;
        set_x_a_test_func_proc(_ml, inst, id, _ppvar, _thread, _nt, *getarg(1));
        hoc_retpushx(_r);
    }
    static double _npy_set_x_a(Prop* _prop) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* _nt;
        _nrn_mechanism_cache_instance _ml_real{_prop};
        auto* const _ml = &_ml_real;
        size_t const id{};
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        _nt = nrn_threads;
        auto inst = make_instance_test_func_proc(_ml_real);
        _r = 1.;
        set_x_a_test_func_proc(_ml, inst, id, _ppvar, _thread, _nt, *getarg(1));
        return(_r);
    }
    static void _hoc_set_a_x(void) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* _nt;
        if (!_prop_id) {
            hoc_execerror("No data for set_a_x_test_func_proc. Requires prior call to setdata_test_func_proc and that the specified mechanism instance still be in existence.", NULL);
        }
        Prop* _local_prop = _extcall_prop;
        _nrn_mechanism_cache_instance _ml_real{_local_prop};
        auto* const _ml = &_ml_real;
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        _nt = nrn_threads;
        auto inst = make_instance_test_func_proc(_ml_real);
        _r = 1.;
        set_a_x_test_func_proc(_ml, inst, id, _ppvar, _thread, _nt);
        hoc_retpushx(_r);
    }
    static double _npy_set_a_x(Prop* _prop) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* _nt;
        _nrn_mechanism_cache_instance _ml_real{_prop};
        auto* const _ml = &_ml_real;
        size_t const id{};
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        _nt = nrn_threads;
        auto inst = make_instance_test_func_proc(_ml_real);
        _r = 1.;
        set_a_x_test_func_proc(_ml, inst, id, _ppvar, _thread, _nt);
        return(_r);
    }
    static void _hoc_x_plus_a(void) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* _nt;
        if (!_prop_id) {
            hoc_execerror("No data for x_plus_a_test_func_proc. Requires prior call to setdata_test_func_proc and that the specified mechanism instance still be in existence.", NULL);
        }
        Prop* _local_prop = _extcall_prop;
        _nrn_mechanism_cache_instance _ml_real{_local_prop};
        auto* const _ml = &_ml_real;
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        _nt = nrn_threads;
        auto inst = make_instance_test_func_proc(_ml_real);
        _r = x_plus_a_test_func_proc(_ml, inst, id, _ppvar, _thread, _nt, *getarg(1));
        hoc_retpushx(_r);
    }
    static double _npy_x_plus_a(Prop* _prop) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* _nt;
        _nrn_mechanism_cache_instance _ml_real{_prop};
        auto* const _ml = &_ml_real;
        size_t const id{};
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        _nt = nrn_threads;
        auto inst = make_instance_test_func_proc(_ml_real);
        _r = x_plus_a_test_func_proc(_ml, inst, id, _ppvar, _thread, _nt, *getarg(1));
        return(_r);
    }


    inline int set_x_42_test_func_proc(_nrn_mechanism_cache_range* _ml, test_func_proc_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
        int ret_set_x_42 = 0;
        auto v = inst.v_unused[id];
        set_x_a_test_func_proc(_ml, inst, id, _ppvar, _thread, _nt, 42.0);
        return ret_set_x_42;
    }


    inline int set_x_a_test_func_proc(_nrn_mechanism_cache_range* _ml, test_func_proc_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double a) {
        int ret_set_x_a = 0;
        auto v = inst.v_unused[id];
        inst.x[id] = a;
        return ret_set_x_a;
    }


    inline int set_a_x_test_func_proc(_nrn_mechanism_cache_range* _ml, test_func_proc_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
        int ret_set_a_x = 0;
        auto v = inst.v_unused[id];
        double a;
        a = inst.x[id];
        return ret_set_a_x;
    }


    inline double x_plus_a_test_func_proc(_nrn_mechanism_cache_range* _ml, test_func_proc_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double a) {
        double ret_x_plus_a = 0.0;
        auto v = inst.v_unused[id];
        ret_x_plus_a = inst.x[id] + a;
        return ret_x_plus_a;
    }


    void nrn_init_test_func_proc(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
        auto inst = make_instance_test_func_proc(_lmr);
        auto node_data = make_node_data_test_func_proc(*_nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* const _ml = &_lmr;
        auto* _thread = _ml_arg->_thread;
        for (int id = 0; id < nodecount; id++) {
            
            int node_id = node_data.nodeindices[id];
            auto* _ppvar = _ml_arg->pdata[id];
            auto v = node_data.node_voltages[node_id];
            inst.v_unused[id] = v;
            set_a_x_test_func_proc(_ml, inst, id, _ppvar, _thread, _nt);
        }
    }


    /** nrn_jacob function */
    static void nrn_jacob_test_func_proc(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
        auto inst = make_instance_test_func_proc(_lmr);
        auto node_data = make_node_data_test_func_proc(*_nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
        }
    }


    static void _initlists() {
    }


    /** register channel with the simulator */
    extern "C" void _func_proc_reg() {
        _initlists();



        register_mech(mechanism_info, nrn_alloc_test_func_proc, nullptr, nullptr, nullptr, nrn_init_test_func_proc, hoc_nrnpointerindex, 1);

        mech_type = nrn_get_mechtype(mechanism_info[1]);
        _nrn_mechanism_register_data_fields(mech_type,
            _nrn_mechanism_field<double>{"x"} /* 0 */,
            _nrn_mechanism_field<double>{"v_unused"} /* 1 */
        );

        hoc_register_prop_size(mech_type, 2, 0);
        hoc_register_var(hoc_scalar_double, hoc_vector_double, hoc_intfunc);
        hoc_register_npy_direct(mech_type, npy_direct_func_proc);
    }
}
