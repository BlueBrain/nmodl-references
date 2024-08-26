/*********************************************************
Model Name      : valence_mod
Filename        : valence.mod
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

static constexpr auto number_of_datum_variables = 2;
static constexpr auto number_of_floating_point_variables = 3;

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
extern void _cvode_abstol(Symbol**, double*, int);
extern Node* nrn_alloc_node_;


namespace neuron {
    #ifndef NRN_PRCELLSTATE
    #define NRN_PRCELLSTATE 0
    #endif


    /** channel information */
    static const char *mechanism_info[] = {
        "7.7.0",
        "valence_mod",
        0,
        "x_valence_mod",
        0,
        0,
        0
    };


    /* NEURON global variables */
    static Symbol* _K_sym;
    static int mech_type;
    static Prop* _extcall_prop;
    /* _prop_id kind of shadows _extcall_prop to allow validity checking. */
    static _nrn_non_owning_id_without_container _prop_id{};
    static int hoc_nrnpointerindex = -1;
    static _nrn_mechanism_std_vector<Datum> _extcall_thread;


    /** all global variables */
    struct valence_mod_Store {
    };
    static_assert(std::is_trivially_copy_constructible_v<valence_mod_Store>);
    static_assert(std::is_trivially_move_constructible_v<valence_mod_Store>);
    static_assert(std::is_trivially_copy_assignable_v<valence_mod_Store>);
    static_assert(std::is_trivially_move_assignable_v<valence_mod_Store>);
    static_assert(std::is_trivially_destructible_v<valence_mod_Store>);
    valence_mod_Store valence_mod_global;
    static std::vector<double> _parameter_defaults = {
    };


    /** all mechanism instance variables and global variables */
    struct valence_mod_Instance  {
        double* x{};
        double* Ki{};
        double* v_unused{};
        const double* const* ion_Ki{};
        const double* const* ion_Ko{};
        valence_mod_Store* global{&valence_mod_global};
    };


    struct valence_mod_NodeData  {
        int const * nodeindices;
        double const * node_voltages;
        double * node_diagonal;
        double * node_rhs;
        int nodecount;
    };


    static valence_mod_Instance make_instance_valence_mod(_nrn_mechanism_cache_range& _lmc) {
        return valence_mod_Instance {
            _lmc.template fpfield_ptr<0>(),
            _lmc.template fpfield_ptr<1>(),
            _lmc.template fpfield_ptr<2>(),
            _lmc.template dptr_field_ptr<0>(),
            _lmc.template dptr_field_ptr<1>()
        };
    }


    static valence_mod_NodeData make_node_data_valence_mod(NrnThread& nt, Memb_list& _ml_arg) {
        return valence_mod_NodeData {
            _ml_arg.nodeindices,
            nt.node_voltage_storage(),
            nt.node_d_storage(),
            nt.node_rhs_storage(),
            _ml_arg.nodecount
        };
    }
    void nrn_destructor_valence_mod(Prop* _prop) {
        Datum* _ppvar = _nrn_mechanism_access_dparam(_prop);
    }


    static void nrn_alloc_valence_mod(Prop* _prop) {
        Datum *_ppvar = nullptr;
        _ppvar = nrn_prop_datum_alloc(mech_type, 2, _prop);
        _nrn_mechanism_access_dparam(_prop) = _ppvar;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const _iml = 0;
        assert(_nrn_mechanism_get_num_vars(_prop) == 3);
        /*initialize range parameters*/
        _nrn_mechanism_access_dparam(_prop) = _ppvar;
        Symbol * K_sym = hoc_lookup("K_ion");
        Prop * K_prop = need_memb(K_sym);
        nrn_promote(K_prop, 1, 0);
        _ppvar[0] = _nrn_mechanism_get_param_handle(K_prop, 1);
        _ppvar[1] = _nrn_mechanism_get_param_handle(K_prop, 2);
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
        {"setdata_valence_mod", _hoc_setdata},
        {nullptr, nullptr}
    };
    static NPyDirectMechFunc npy_direct_func_proc[] = {
        {nullptr, nullptr}
    };


    void nrn_init_valence_mod(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _type};
        auto inst = make_instance_valence_mod(_lmc);
        auto node_data = make_node_data_valence_mod(*nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* _thread = _ml_arg->_thread;
        for (int id = 0; id < nodecount; id++) {
            auto* _ppvar = _ml_arg->pdata[id];
            int node_id = node_data.nodeindices[id];
            auto v = node_data.node_voltages[node_id];
            inst.v_unused[id] = v;
            inst.Ki[id] = (*inst.ion_Ki[id]);
            inst.x[id] = inst.Ki[id];
        }
    }


    static void nrn_jacob_valence_mod(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _type};
        auto inst = make_instance_valence_mod(_lmc);
        auto node_data = make_node_data_valence_mod(*nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
        }
    }


    static void _initlists() {
    }


    /** register channel with the simulator */
    extern "C" void _valence_reg() {
        _initlists();

        ion_reg("K", 222);

        _K_sym = hoc_lookup("K_ion");

        register_mech(mechanism_info, nrn_alloc_valence_mod, nullptr, nullptr, nullptr, nrn_init_valence_mod, hoc_nrnpointerindex, 1);

        mech_type = nrn_get_mechtype(mechanism_info[1]);
        hoc_register_parm_default(mech_type, &_parameter_defaults);
        _nrn_mechanism_register_data_fields(mech_type,
            _nrn_mechanism_field<double>{"x"} /* 0 */,
            _nrn_mechanism_field<double>{"Ki"} /* 1 */,
            _nrn_mechanism_field<double>{"v_unused"} /* 2 */,
            _nrn_mechanism_field<double*>{"ion_Ki", "K_ion"} /* 0 */,
            _nrn_mechanism_field<double*>{"ion_Ko", "K_ion"} /* 1 */
        );

        hoc_register_prop_size(mech_type, 3, 2);
        hoc_register_dparam_semantics(mech_type, 0, "K_ion");
        hoc_register_dparam_semantics(mech_type, 1, "K_ion");
        hoc_register_var(hoc_scalar_double, hoc_vector_double, hoc_intfunc);
        hoc_register_npy_direct(mech_type, npy_direct_func_proc);
    }
}
