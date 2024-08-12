/*********************************************************
Model Name      : internal_function
Filename        : internal_function.mod
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

/* NEURON global macro definitions */
/* VECTORIZED */
#define NRN_VECTORIZED 1

static constexpr auto number_of_datum_variables = 0;
static constexpr auto number_of_floating_point_variables = 1;

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
extern Node* nrn_alloc_node_;


namespace neuron {
    #ifndef NRN_PRCELLSTATE
    #define NRN_PRCELLSTATE 0
    #endif


    /** channel information */
    static const char *mechanism_info[] = {
        "7.7.0",
        "internal_function",
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
    struct internal_function_Store {
    };
    static_assert(std::is_trivially_copy_constructible_v<internal_function_Store>);
    static_assert(std::is_trivially_move_constructible_v<internal_function_Store>);
    static_assert(std::is_trivially_copy_assignable_v<internal_function_Store>);
    static_assert(std::is_trivially_move_assignable_v<internal_function_Store>);
    static_assert(std::is_trivially_destructible_v<internal_function_Store>);
    internal_function_Store internal_function_global;
    static std::vector<double> _parameter_defaults = {
    };


    /** all mechanism instance variables and global variables */
    struct internal_function_Instance  {
        double* v_unused{};
        internal_function_Store* global{&internal_function_global};
    };


    struct internal_function_NodeData  {
        int const * nodeindices;
        double const * node_voltages;
        double * node_diagonal;
        double * node_rhs;
        int nodecount;
    };


    static internal_function_Instance make_instance_internal_function(_nrn_mechanism_cache_range& _lmc) {
        return internal_function_Instance {
            _lmc.template fpfield_ptr<0>()
        };
    }


    static internal_function_NodeData make_node_data_internal_function(NrnThread& nt, Memb_list& _ml_arg) {
        return internal_function_NodeData {
            _ml_arg.nodeindices,
            nt.node_voltage_storage(),
            nt.node_d_storage(),
            nt.node_rhs_storage(),
            _ml_arg.nodecount
        };
    }


    static void nrn_alloc_internal_function(Prop* _prop) {
        Datum *_ppvar = nullptr;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const _iml = 0;
        assert(_nrn_mechanism_get_num_vars(_prop) == 1);
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
    inline double f_internal_function(_nrn_mechanism_cache_range& _lmc, internal_function_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt);
    inline double g_internal_function(_nrn_mechanism_cache_range& _lmc, internal_function_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt);


    /** connect global (scalar) variables to hoc -- */
    static DoubScal hoc_scalar_double[] = {
        {nullptr, nullptr}
    };


    /** connect global (array) variables to hoc -- */
    static DoubVec hoc_vector_double[] = {
        {nullptr, nullptr, 0}
    };


    /* declaration of user functions */
    static void _hoc_f(void);
    static void _hoc_g(void);
    static double _npy_f(Prop*);
    static double _npy_g(Prop*);


    /* connect user functions to hoc names */
    static VoidFunc hoc_intfunc[] = {
        {"setdata_internal_function", _hoc_setdata},
        {"f_internal_function", _hoc_f},
        {"g_internal_function", _hoc_g},
        {nullptr, nullptr}
    };
    static NPyDirectMechFunc npy_direct_func_proc[] = {
        {"f", _npy_f},
        {"g", _npy_g},
        {nullptr, nullptr}
    };
    static void _hoc_f(void) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        Prop* _local_prop = _prop_id ? _extcall_prop : nullptr;
        _nrn_mechanism_cache_instance _lmc{_local_prop};
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_internal_function(_lmc);
        _r = f_internal_function(_lmc, inst, id, _ppvar, _thread, nt);
        hoc_retpushx(_r);
    }
    static double _npy_f(Prop* _prop) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id{};
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_internal_function(_lmc);
        _r = f_internal_function(_lmc, inst, id, _ppvar, _thread, nt);
        return(_r);
    }
    static void _hoc_g(void) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        Prop* _local_prop = _prop_id ? _extcall_prop : nullptr;
        _nrn_mechanism_cache_instance _lmc{_local_prop};
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_internal_function(_lmc);
        _r = g_internal_function(_lmc, inst, id, _ppvar, _thread, nt);
        hoc_retpushx(_r);
    }
    static double _npy_g(Prop* _prop) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id{};
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_internal_function(_lmc);
        _r = g_internal_function(_lmc, inst, id, _ppvar, _thread, nt);
        return(_r);
    }


    inline double f_internal_function(_nrn_mechanism_cache_range& _lmc, internal_function_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt) {
        double ret_f = 0.0;
        auto v = inst.v_unused[id];
        // Setup for VERBATIM
        #define g g_internal_function
        #define t nt->_t
        #define _internalthreadargs_ _lmc, inst, id, _ppvar, _thread, nt
        // Begin VERBATIM
        
          return g(_internalthreadargs_);
        // End VERBATIM
        #undef g
        #undef t
        #undef _internalthreadargs_
        // End of cleanup for VERBATIM

        return ret_f;
    }


    inline double g_internal_function(_nrn_mechanism_cache_range& _lmc, internal_function_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt) {
        double ret_g = 0.0;
        auto v = inst.v_unused[id];
        ret_g = 42.0;
        return ret_g;
    }


    void nrn_init_internal_function(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _type};
        auto inst = make_instance_internal_function(_lmc);
        auto node_data = make_node_data_internal_function(*nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* _thread = _ml_arg->_thread;
        for (int id = 0; id < nodecount; id++) {
            auto* _ppvar = _ml_arg->pdata[id];
            int node_id = node_data.nodeindices[id];
            auto v = node_data.node_voltages[node_id];
            inst.v_unused[id] = v;
        }
    }


    static void nrn_jacob_internal_function(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _type};
        auto inst = make_instance_internal_function(_lmc);
        auto node_data = make_node_data_internal_function(*nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
        }
    }


    static void _initlists() {
    }


    /** register channel with the simulator */
    extern "C" void _internal_function_reg() {
        _initlists();

        register_mech(mechanism_info, nrn_alloc_internal_function, nullptr, nullptr, nullptr, nrn_init_internal_function, hoc_nrnpointerindex, 1);

        mech_type = nrn_get_mechtype(mechanism_info[1]);
        hoc_register_parm_default(mech_type, &_parameter_defaults);
        _nrn_mechanism_register_data_fields(mech_type,
            _nrn_mechanism_field<double>{"v_unused"} /* 0 */
        );

        hoc_register_prop_size(mech_type, 1, 0);
        hoc_register_var(hoc_scalar_double, hoc_vector_double, hoc_intfunc);
        hoc_register_npy_direct(mech_type, npy_direct_func_proc);
    }
}
