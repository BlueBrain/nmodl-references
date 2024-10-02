/*********************************************************
Model Name      : default_parameter
Filename        : default_parameter.mod
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
extern void _cvode_abstol(Symbol**, double*, int);
extern Node* nrn_alloc_node_;


namespace neuron {
    #ifndef NRN_PRCELLSTATE
    #define NRN_PRCELLSTATE 0
    #endif


    /** channel information */
    static const char *mechanism_info[] = {
        "7.7.0",
        "default_parameter",
        "x_default_parameter",
        "y_default_parameter",
        "z_default_parameter",
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
    static _nrn_mechanism_std_vector<Datum> _extcall_thread;


    /** all global variables */
    struct default_parameter_Store {
        double a{0};
        double b{0.1};
    };
    static_assert(std::is_trivially_copy_constructible_v<default_parameter_Store>);
    static_assert(std::is_trivially_move_constructible_v<default_parameter_Store>);
    static_assert(std::is_trivially_copy_assignable_v<default_parameter_Store>);
    static_assert(std::is_trivially_move_assignable_v<default_parameter_Store>);
    static_assert(std::is_trivially_destructible_v<default_parameter_Store>);
    default_parameter_Store default_parameter_global;
    auto a_default_parameter() -> std::decay<decltype(default_parameter_global.a)>::type  {
        return default_parameter_global.a;
    }
    auto b_default_parameter() -> std::decay<decltype(default_parameter_global.b)>::type  {
        return default_parameter_global.b;
    }

    static std::vector<double> _parameter_defaults = {
        0 /* x */,
        2.1 /* y */,
        0 /* z */
    };


    /** all mechanism instance variables and global variables */
    struct default_parameter_Instance  {
        double* x{};
        double* y{};
        double* z{};
        double* v_unused{};
        default_parameter_Store* global{&default_parameter_global};
    };


    struct default_parameter_NodeData  {
        int const * nodeindices;
        double const * node_voltages;
        double * node_diagonal;
        double * node_rhs;
        int nodecount;
    };


    static default_parameter_Instance make_instance_default_parameter(_nrn_mechanism_cache_range& _lmc) {
        return default_parameter_Instance {
            _lmc.template fpfield_ptr<0>(),
            _lmc.template fpfield_ptr<1>(),
            _lmc.template fpfield_ptr<2>(),
            _lmc.template fpfield_ptr<3>()
        };
    }


    static default_parameter_NodeData make_node_data_default_parameter(NrnThread& nt, Memb_list& _ml_arg) {
        return default_parameter_NodeData {
            _ml_arg.nodeindices,
            nt.node_voltage_storage(),
            nt.node_d_storage(),
            nt.node_rhs_storage(),
            _ml_arg.nodecount
        };
    }
    static default_parameter_NodeData make_node_data_default_parameter(Prop * _prop) {
        static std::vector<int> node_index{0};
        Node* _node = _nrn_mechanism_access_node(_prop);
        return default_parameter_NodeData {
            node_index.data(),
            &_nrn_mechanism_access_voltage(_node),
            &_nrn_mechanism_access_d(_node),
            &_nrn_mechanism_access_rhs(_node),
            1
        };
    }

    void nrn_destructor_default_parameter(Prop* prop);


    static void nrn_alloc_default_parameter(Prop* _prop) {
        Datum *_ppvar = nullptr;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const _iml = 0;
        assert(_nrn_mechanism_get_num_vars(_prop) == 4);
        /*initialize range parameters*/
        _lmc.template fpfield<0>(_iml) = _parameter_defaults[0]; /* x */
        _lmc.template fpfield<1>(_iml) = _parameter_defaults[1]; /* y */
        _lmc.template fpfield<2>(_iml) = _parameter_defaults[2]; /* z */
    }


    /* Mechanism procedures and functions */
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


    /** connect global (scalar) variables to hoc -- */
    static DoubScal hoc_scalar_double[] = {
        {"a_default_parameter", &default_parameter_global.a},
        {"b_default_parameter", &default_parameter_global.b},
        {nullptr, nullptr}
    };


    /** connect global (array) variables to hoc -- */
    static DoubVec hoc_vector_double[] = {
        {nullptr, nullptr, 0}
    };


    /* declaration of user functions */


    /* connect user functions to hoc names */
    static VoidFunc hoc_intfunc[] = {
        {"setdata_default_parameter", _hoc_setdata},
        {nullptr, nullptr}
    };
    static NPyDirectMechFunc npy_direct_func_proc[] = {
        {nullptr, nullptr}
    };


    void nrn_init_default_parameter(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _type};
        auto inst = make_instance_default_parameter(_lmc);
        auto node_data = make_node_data_default_parameter(*nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* _thread = _ml_arg->_thread;
        for (int id = 0; id < nodecount; id++) {
            auto* _ppvar = _ml_arg->pdata[id];
            int node_id = node_data.nodeindices[id];
            auto v = node_data.node_voltages[node_id];
        }
    }


    static void nrn_jacob_default_parameter(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _type};
        auto inst = make_instance_default_parameter(_lmc);
        auto node_data = make_node_data_default_parameter(*nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
        }
    }
    void nrn_destructor_default_parameter(Prop* prop) {
        Datum* _ppvar = _nrn_mechanism_access_dparam(prop);
        _nrn_mechanism_cache_instance _lmc{prop};
        const size_t id = 0;
        auto inst = make_instance_default_parameter(_lmc);
        auto node_data = make_node_data_default_parameter(prop);

    }


    static void _initlists() {
    }


    /** register channel with the simulator */
    extern "C" void _default_parameter_reg() {
        _initlists();

        register_mech(mechanism_info, nrn_alloc_default_parameter, nullptr, nullptr, nullptr, nrn_init_default_parameter, -1, 1);

        mech_type = nrn_get_mechtype(mechanism_info[1]);
        hoc_register_parm_default(mech_type, &_parameter_defaults);
        _nrn_mechanism_register_data_fields(mech_type,
            _nrn_mechanism_field<double>{"x"} /* 0 */,
            _nrn_mechanism_field<double>{"y"} /* 1 */,
            _nrn_mechanism_field<double>{"z"} /* 2 */,
            _nrn_mechanism_field<double>{"v_unused"} /* 3 */
        );

        hoc_register_prop_size(mech_type, 4, 0);
        hoc_register_var(hoc_scalar_double, hoc_vector_double, hoc_intfunc);
        hoc_register_npy_direct(mech_type, npy_direct_func_proc);
    }
}
