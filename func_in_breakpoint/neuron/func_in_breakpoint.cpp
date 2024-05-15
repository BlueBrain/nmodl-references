/*********************************************************
Model Name      : func_in_breakpoint
Filename        : func_in_breakpoint.mod
NMODL Version   : 7.7.0
Vectorized      : true
Threadsafe      : true
Created         : DATE
Simulator       : NEURON
Backend         : C++ (api-compatibility)
NMODL Compiler  : VERSION
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
        "func_in_breakpoint",
        0,
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
    struct func_in_breakpoint_Store {
        double c{1};
    };
    static_assert(std::is_trivially_copy_constructible_v<func_in_breakpoint_Store>);
    static_assert(std::is_trivially_move_constructible_v<func_in_breakpoint_Store>);
    static_assert(std::is_trivially_copy_assignable_v<func_in_breakpoint_Store>);
    static_assert(std::is_trivially_move_assignable_v<func_in_breakpoint_Store>);
    static_assert(std::is_trivially_destructible_v<func_in_breakpoint_Store>);
    func_in_breakpoint_Store func_in_breakpoint_global;


    /** all mechanism instance variables and global variables */
    struct func_in_breakpoint_Instance  {
        double* v_unused{};
        double* g_unused{};
        func_in_breakpoint_Store* global{&func_in_breakpoint_global};
    };


    struct func_in_breakpoint_NodeData  {
        int const * nodeindices;
        double const * node_voltages;
        double * node_diagonal;
        double * node_rhs;
        int nodecount;
    };


    static func_in_breakpoint_Instance make_instance_func_in_breakpoint(_nrn_mechanism_cache_range& _ml) {
        return func_in_breakpoint_Instance {
            _ml.template fpfield_ptr<0>(),
            _ml.template fpfield_ptr<1>()
        };
    }


    static func_in_breakpoint_NodeData make_node_data_func_in_breakpoint(NrnThread& _nt, Memb_list& _ml_arg) {
        return func_in_breakpoint_NodeData {
            _ml_arg.nodeindices,
            _nt.node_voltage_storage(),
            _nt.node_d_storage(),
            _nt.node_rhs_storage(),
            _ml_arg.nodecount
        };
    }


    static void nrn_alloc_func_in_breakpoint(Prop* _prop) {
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
    inline int func_func_in_breakpoint(_nrn_mechanism_cache_range* _ml, func_in_breakpoint_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* _nt);
    inline int func_with_v_func_in_breakpoint(_nrn_mechanism_cache_range* _ml, func_in_breakpoint_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double v);
    inline int func_with_other_func_in_breakpoint(_nrn_mechanism_cache_range* _ml, func_in_breakpoint_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double q);


    /** connect global (scalar) variables to hoc -- */
    static DoubScal hoc_scalar_double[] = {
        {"c_func_in_breakpoint", &func_in_breakpoint_global.c},
        {nullptr, nullptr}
    };


    /** connect global (array) variables to hoc -- */
    static DoubVec hoc_vector_double[] = {
        {nullptr, nullptr, 0}
    };


    /* declaration of user functions */
    static void _hoc_func(void);
    static void _hoc_func_with_v(void);
    static void _hoc_func_with_other(void);
    static double _npy_func(Prop*);
    static double _npy_func_with_v(Prop*);
    static double _npy_func_with_other(Prop*);


    /* connect user functions to hoc names */
    static VoidFunc hoc_intfunc[] = {
        {"setdata_func_in_breakpoint", _hoc_setdata},
        {"func_func_in_breakpoint", _hoc_func},
        {"func_with_v_func_in_breakpoint", _hoc_func_with_v},
        {"func_with_other_func_in_breakpoint", _hoc_func_with_other},
        {0, 0}
    };
    static NPyDirectMechFunc npy_direct_func_proc[] = {
        {"func", _npy_func},
        {"func_with_v", _npy_func_with_v},
        {"func_with_other", _npy_func_with_other},
    };
    static void _hoc_func(void) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* _nt;
        Prop* _local_prop = _prop_id ? _extcall_prop : nullptr;
        _nrn_mechanism_cache_instance _ml_real{_local_prop};
        auto* const _ml = &_ml_real;
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        _nt = nrn_threads;
        auto inst = make_instance_func_in_breakpoint(_ml_real);
        _r = 1.;
        func_func_in_breakpoint(_ml, inst, id, _ppvar, _thread, _nt);
        hoc_retpushx(_r);
    }
    static double _npy_func(Prop* _prop) {
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
        auto inst = make_instance_func_in_breakpoint(_ml_real);
        _r = 1.;
        func_func_in_breakpoint(_ml, inst, id, _ppvar, _thread, _nt);
        return(_r);
    }
    static void _hoc_func_with_v(void) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* _nt;
        Prop* _local_prop = _prop_id ? _extcall_prop : nullptr;
        _nrn_mechanism_cache_instance _ml_real{_local_prop};
        auto* const _ml = &_ml_real;
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        _nt = nrn_threads;
        auto inst = make_instance_func_in_breakpoint(_ml_real);
        _r = 1.;
        func_with_v_func_in_breakpoint(_ml, inst, id, _ppvar, _thread, _nt, *getarg(1));
        hoc_retpushx(_r);
    }
    static double _npy_func_with_v(Prop* _prop) {
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
        auto inst = make_instance_func_in_breakpoint(_ml_real);
        _r = 1.;
        func_with_v_func_in_breakpoint(_ml, inst, id, _ppvar, _thread, _nt, *getarg(1));
        return(_r);
    }
    static void _hoc_func_with_other(void) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* _nt;
        Prop* _local_prop = _prop_id ? _extcall_prop : nullptr;
        _nrn_mechanism_cache_instance _ml_real{_local_prop};
        auto* const _ml = &_ml_real;
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        _nt = nrn_threads;
        auto inst = make_instance_func_in_breakpoint(_ml_real);
        _r = 1.;
        func_with_other_func_in_breakpoint(_ml, inst, id, _ppvar, _thread, _nt, *getarg(1));
        hoc_retpushx(_r);
    }
    static double _npy_func_with_other(Prop* _prop) {
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
        auto inst = make_instance_func_in_breakpoint(_ml_real);
        _r = 1.;
        func_with_other_func_in_breakpoint(_ml, inst, id, _ppvar, _thread, _nt, *getarg(1));
        return(_r);
    }


    inline int func_func_in_breakpoint(_nrn_mechanism_cache_range* _ml, func_in_breakpoint_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
        int ret_func = 0;
        auto v = inst.v_unused[id];
        return ret_func;
    }


    inline int func_with_v_func_in_breakpoint(_nrn_mechanism_cache_range* _ml, func_in_breakpoint_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double v) {
        int ret_func_with_v = 0;
        return ret_func_with_v;
    }


    inline int func_with_other_func_in_breakpoint(_nrn_mechanism_cache_range* _ml, func_in_breakpoint_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double q) {
        int ret_func_with_other = 0;
        auto v = inst.v_unused[id];
        return ret_func_with_other;
    }


    void nrn_init_func_in_breakpoint(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
        auto inst = make_instance_func_in_breakpoint(_lmr);
        auto node_data = make_node_data_func_in_breakpoint(*_nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* const _ml = &_lmr;
        auto* _thread = _ml_arg->_thread;
        for (int id = 0; id < nodecount; id++) {
            
            int node_id = node_data.nodeindices[id];
            auto* _ppvar = _ml_arg->pdata[id];
            auto v = node_data.node_voltages[node_id];
            inst.v_unused[id] = v;
        }
    }


    void nrn_state_func_in_breakpoint(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
        auto inst = make_instance_func_in_breakpoint(_lmr);
        auto node_data = make_node_data_func_in_breakpoint(*_nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* const _ml = &_lmr;
        auto* _thread = _ml_arg->_thread;
        for (int id = 0; id < nodecount; id++) {
            
            int node_id = node_data.nodeindices[id];
            auto* _ppvar = _ml_arg->pdata[id];
            auto v = node_data.node_voltages[node_id];
            func_func_in_breakpoint(_ml, inst, id, _ppvar, _thread, _nt);
            func_with_v_func_in_breakpoint(_ml, inst, id, _ppvar, _thread, _nt, v);
            func_with_other_func_in_breakpoint(_ml, inst, id, _ppvar, _thread, _nt, inst.global->c);
        }
    }


    /** nrn_jacob function */
    static void nrn_jacob_func_in_breakpoint(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
        auto inst = make_instance_func_in_breakpoint(_lmr);
        auto node_data = make_node_data_func_in_breakpoint(*_nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
            // set conductances properly
            int node_id = node_data.nodeindices[id];
            node_data.node_diagonal[node_id] += inst.g_unused[id];
        }
    }


    static void _initlists() {
    }


    /** register channel with the simulator */
    extern "C" void _func_in_breakpoint_reg() {
        _initlists();



        register_mech(mechanism_info, nrn_alloc_func_in_breakpoint, nullptr, nrn_jacob_func_in_breakpoint, nrn_state_func_in_breakpoint, nrn_init_func_in_breakpoint, hoc_nrnpointerindex, 1);

        mech_type = nrn_get_mechtype(mechanism_info[1]);
        _nrn_mechanism_register_data_fields(mech_type,
            _nrn_mechanism_field<double>{"v_unused"} /* 0 */,
            _nrn_mechanism_field<double>{"g_unused"} /* 1 */
        );

        hoc_register_prop_size(mech_type, 2, 0);
        hoc_register_var(hoc_scalar_double, hoc_vector_double, hoc_intfunc);
        hoc_register_npy_direct(mech_type, npy_direct_func_proc);
    }
}
