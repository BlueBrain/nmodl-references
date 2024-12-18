/*********************************************************
Model Name      : lin
Filename        : lin.mod
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
#include "nmodlmutex.h"
#include "nrniv_mf.h"
#include "section_fwd.hpp"

/* NEURON global macro definitions */
/* VECTORIZED */
#define NRN_VECTORIZED 1

static constexpr auto number_of_datum_variables = 0;
static constexpr auto number_of_floating_point_variables = 5;

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
        "lin",
        0,
        0,
        "xx_lin",
        "yy_lin",
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
    struct lin_Store {
        double a{2};
        double b{3};
        double c{4};
        double d{5};
        double xx0{0};
        double yy0{0};
    };
    static_assert(std::is_trivially_copy_constructible_v<lin_Store>);
    static_assert(std::is_trivially_move_constructible_v<lin_Store>);
    static_assert(std::is_trivially_copy_assignable_v<lin_Store>);
    static_assert(std::is_trivially_move_assignable_v<lin_Store>);
    static_assert(std::is_trivially_destructible_v<lin_Store>);
    static lin_Store lin_global;
    auto a_lin() -> std::decay<decltype(lin_global.a)>::type  {
        return lin_global.a;
    }
    auto b_lin() -> std::decay<decltype(lin_global.b)>::type  {
        return lin_global.b;
    }
    auto c_lin() -> std::decay<decltype(lin_global.c)>::type  {
        return lin_global.c;
    }
    auto d_lin() -> std::decay<decltype(lin_global.d)>::type  {
        return lin_global.d;
    }
    auto xx0_lin() -> std::decay<decltype(lin_global.xx0)>::type  {
        return lin_global.xx0;
    }
    auto yy0_lin() -> std::decay<decltype(lin_global.yy0)>::type  {
        return lin_global.yy0;
    }

    static std::vector<double> _parameter_defaults = {
    };


    /** all mechanism instance variables and global variables */
    struct lin_Instance  {
        double* xx{};
        double* yy{};
        double* Dxx{};
        double* Dyy{};
        double* v_unused{};
        lin_Store* global{&lin_global};
    };


    struct lin_NodeData  {
        int const * nodeindices;
        double const * node_voltages;
        double * node_diagonal;
        double * node_rhs;
        int nodecount;
    };


    static lin_Instance make_instance_lin(_nrn_mechanism_cache_range* _lmc) {
        if(_lmc == nullptr) {
            return lin_Instance();
        }

        return lin_Instance {
            _lmc->template fpfield_ptr<0>(),
            _lmc->template fpfield_ptr<1>(),
            _lmc->template fpfield_ptr<2>(),
            _lmc->template fpfield_ptr<3>(),
            _lmc->template fpfield_ptr<4>()
        };
    }


    static lin_NodeData make_node_data_lin(NrnThread& nt, Memb_list& _ml_arg) {
        return lin_NodeData {
            _ml_arg.nodeindices,
            nt.node_voltage_storage(),
            nt.node_d_storage(),
            nt.node_rhs_storage(),
            _ml_arg.nodecount
        };
    }
    static lin_NodeData make_node_data_lin(Prop * _prop) {
        if(!_prop) {
            return lin_NodeData();
        }

        static std::vector<int> node_index{0};
        Node* _node = _nrn_mechanism_access_node(_prop);
        return lin_NodeData {
            node_index.data(),
            &_nrn_mechanism_access_voltage(_node),
            &_nrn_mechanism_access_d(_node),
            &_nrn_mechanism_access_rhs(_node),
            1
        };
    }

    static void nrn_destructor_lin(Prop* prop);


    static void nrn_alloc_lin(Prop* _prop) {
        Datum *_ppvar = nullptr;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const _iml = 0;
        assert(_nrn_mechanism_get_num_vars(_prop) == 5);
        /*initialize range parameters*/
    }


    /* Mechanism procedures and functions */
    static void _apply_diffusion_function(ldifusfunc2_t _f, const _nrn_model_sorted_token& _sorted_token, NrnThread& _nt) {
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


    /** connect global (scalar) variables to hoc -- */
    static DoubScal hoc_scalar_double[] = {
        {"a_lin", &lin_global.a},
        {"b_lin", &lin_global.b},
        {"c_lin", &lin_global.c},
        {"d_lin", &lin_global.d},
        {nullptr, nullptr}
    };


    /** connect global (array) variables to hoc -- */
    static DoubVec hoc_vector_double[] = {
        {nullptr, nullptr, 0}
    };


    /* declaration of user functions */


    /* connect user functions to hoc names */
    static VoidFunc hoc_intfunc[] = {
        {"setdata_lin", _hoc_setdata},
        {nullptr, nullptr}
    };
    static NPyDirectMechFunc npy_direct_func_proc[] = {
        {nullptr, nullptr}
    };


    static void nrn_init_lin(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_lin(&_lmc);
        auto node_data = make_node_data_lin(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
            auto* _ppvar = _ml_arg->pdata[id];
            int node_id = node_data.nodeindices[id];
            inst.v_unused[id] = node_data.node_voltages[node_id];
            inst.xx[id] = inst.global->xx0;
            inst.yy[id] = inst.global->yy0;
                        inst.xx[id] = 0.0;
            inst.yy[id] = 0.0;

        }
    }


    static void nrn_jacob_lin(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_lin(&_lmc);
        auto node_data = make_node_data_lin(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
        }
    }
    static void nrn_destructor_lin(Prop* prop) {
        Datum* _ppvar = _nrn_mechanism_access_dparam(prop);
        _nrn_mechanism_cache_instance _lmc{prop};
        const size_t id = 0;
        auto inst = make_instance_lin(prop ? &_lmc : nullptr);
        auto node_data = make_node_data_lin(prop);

    }


    static void _initlists() {
    }


    extern "C" void _lin_reg() {
        _initlists();

        register_mech(mechanism_info, nrn_alloc_lin, nullptr, nullptr, nullptr, nrn_init_lin, -1, 1);

        mech_type = nrn_get_mechtype(mechanism_info[1]);
        hoc_register_parm_default(mech_type, &_parameter_defaults);
        _nrn_mechanism_register_data_fields(mech_type,
            _nrn_mechanism_field<double>{"xx"} /* 0 */,
            _nrn_mechanism_field<double>{"yy"} /* 1 */,
            _nrn_mechanism_field<double>{"Dxx"} /* 2 */,
            _nrn_mechanism_field<double>{"Dyy"} /* 3 */,
            _nrn_mechanism_field<double>{"v_unused"} /* 4 */
        );

        hoc_register_prop_size(mech_type, 5, 0);
        hoc_register_var(hoc_scalar_double, hoc_vector_double, hoc_intfunc);
        hoc_register_npy_direct(mech_type, npy_direct_func_proc);
    }
}
