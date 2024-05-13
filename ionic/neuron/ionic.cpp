/*********************************************************
Model Name      : ionic
Filename        : ionic.mod
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

static constexpr auto number_of_datum_variables = 2;
static constexpr auto number_of_floating_point_variables = 4;

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
        "ionic",
        0,
        0,
        0,
        0
    };


    /* NEURON global variables */
    static Symbol* _na_sym;
    static int mech_type;
    static Prop* _extcall_prop;
    /* _prop_id kind of shadows _extcall_prop to allow validity checking. */
    static _nrn_non_owning_id_without_container _prop_id{};
    static int hoc_nrnpointerindex = -1;
    static _nrn_mechanism_std_vector<Datum> _extcall_thread;


    /** all global variables */
    struct ionic_Store {
    };
    static_assert(std::is_trivially_copy_constructible_v<ionic_Store>);
    static_assert(std::is_trivially_move_constructible_v<ionic_Store>);
    static_assert(std::is_trivially_copy_assignable_v<ionic_Store>);
    static_assert(std::is_trivially_move_assignable_v<ionic_Store>);
    static_assert(std::is_trivially_destructible_v<ionic_Store>);
    ionic_Store ionic_global;


    /** all mechanism instance variables and global variables */
    struct ionic_Instance  {
        double* ina{};
        double* ena{};
        double* v_unused{};
        double* g_unused{};
        const double* const* ion_ina{};
        double* const* ion_ena{};
        ionic_Store* global{&ionic_global};
    };


    struct ionic_NodeData  {
        int const * nodeindices;
        double const * node_voltages;
        double * node_diagonal;
        double * node_rhs;
        int nodecount;
    };


    static ionic_Instance make_instance_ionic(_nrn_mechanism_cache_range& _ml) {
        return ionic_Instance {
            _ml.template fpfield_ptr<0>(),
            _ml.template fpfield_ptr<1>(),
            _ml.template fpfield_ptr<2>(),
            _ml.template fpfield_ptr<3>(),
            _ml.template dptr_field_ptr<0>(),
            _ml.template dptr_field_ptr<1>()
        };
    }


    static ionic_NodeData make_node_data_ionic(NrnThread& _nt, Memb_list& _ml_arg) {
        return ionic_NodeData {
            _ml_arg.nodeindices,
            _nt.node_voltage_storage(),
            _nt.node_d_storage(),
            _nt.node_rhs_storage(),
            _ml_arg.nodecount
        };
    }


    static void nrn_alloc_ionic(Prop* _prop) {
        Prop *prop_ion{};
        Datum *_ppvar{};
        _ppvar = nrn_prop_datum_alloc(mech_type, 2, _prop);
        _nrn_mechanism_access_dparam(_prop) = _ppvar;
        _nrn_mechanism_cache_instance _ml_real{_prop};
        auto* const _ml = &_ml_real;
        size_t const _iml{};
        assert(_nrn_mechanism_get_num_vars(_prop) == 4);
        /*initialize range parameters*/
        _nrn_mechanism_access_dparam(_prop) = _ppvar;
        Symbol * na_sym = hoc_lookup("na_ion");
        Prop * na_prop = need_memb(na_sym);
        _ppvar[0] = _nrn_mechanism_get_param_handle(na_prop, 3);
        _ppvar[1] = _nrn_mechanism_get_param_handle(na_prop, 0);
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


    /** connect global (scalar) variables to hoc -- */
    static DoubScal hoc_scalar_double[] = {
        {nullptr, nullptr}
    };


    /** connect global (array) variables to hoc -- */
    static DoubVec hoc_vector_double[] = {
        {nullptr, nullptr, 0}
    };


    /* declaration of user functions */


    /* connect user functions to hoc names */
    static VoidFunc hoc_intfunc[] = {
        {"setdata_ionic", _hoc_setdata},
        {0, 0}
    };
    static NPyDirectMechFunc npy_direct_func_proc[] = {
    };


    void nrn_init_ionic(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
        auto inst = make_instance_ionic(_lmr);
        auto node_data = make_node_data_ionic(*_nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* const _ml = &_lmr;
        auto* _thread = _ml_arg->_thread;
        for (int id = 0; id < nodecount; id++) {
            
            int node_id = node_data.nodeindices[id];
            auto* _ppvar = _ml_arg->pdata[id];
            auto v = node_data.node_voltages[node_id];
            inst.v_unused[id] = v;
            inst.ina[id] = (*inst.ion_ina[id]);
            (*inst.ion_ena[id]) = inst.ena[id];
        }
    }


    void nrn_state_ionic(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
        auto inst = make_instance_ionic(_lmr);
        auto node_data = make_node_data_ionic(*_nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* const _ml = &_lmr;
        auto* _thread = _ml_arg->_thread;
        for (int id = 0; id < nodecount; id++) {
            
            int node_id = node_data.nodeindices[id];
            auto* _ppvar = _ml_arg->pdata[id];
            auto v = node_data.node_voltages[node_id];
            inst.ina[id] = (*inst.ion_ina[id]);
            inst.ena[id] = 42.0;
            (*inst.ion_ena[id]) = inst.ena[id];
        }
    }


    /** nrn_jacob function */
    static void nrn_jacob_ionic(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
        auto inst = make_instance_ionic(_lmr);
        auto node_data = make_node_data_ionic(*_nt, *_ml_arg);
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
    extern "C" void _ionic_reg() {
        _initlists();

        ion_reg("na", -10000.);

        _na_sym = hoc_lookup("na_ion");

        register_mech(mechanism_info, nrn_alloc_ionic, nullptr, nrn_jacob_ionic, nrn_state_ionic, nrn_init_ionic, hoc_nrnpointerindex, 1);

        mech_type = nrn_get_mechtype(mechanism_info[1]);
        _nrn_mechanism_register_data_fields(mech_type,
            _nrn_mechanism_field<double>{"ina"} /* 0 */,
            _nrn_mechanism_field<double>{"ena"} /* 1 */,
            _nrn_mechanism_field<double>{"v_unused"} /* 2 */,
            _nrn_mechanism_field<double>{"g_unused"} /* 3 */,
            _nrn_mechanism_field<double*>{"ion_ina", "na_ion"} /* 0 */,
            _nrn_mechanism_field<double*>{"ion_ena", "na_ion"} /* 1 */
        );

        hoc_register_prop_size(mech_type, 4, 2);
        hoc_register_dparam_semantics(mech_type, 0, "na_ion");
        hoc_register_dparam_semantics(mech_type, 1, "na_ion");
        hoc_register_var(hoc_scalar_double, hoc_vector_double, hoc_intfunc);
        hoc_register_npy_direct(mech_type, npy_direct_func_proc);
    }
}
