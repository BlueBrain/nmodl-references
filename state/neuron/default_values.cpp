/*********************************************************
Model Name      : default_values
Filename        : default_values.mod
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
static constexpr auto number_of_floating_point_variables = 11;

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
        "default_values",
        0,
        0,
        "X_default_values",
        "Y_default_values",
        "Z_default_values",
        "A_default_values[3]",
        "B_default_values[2]",
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
    struct default_values_Store {
        double X0{2};
        double Z0{3};
        double A0{4};
        double B0{5};
        double Y0{0};
    };
    static_assert(std::is_trivially_copy_constructible_v<default_values_Store>);
    static_assert(std::is_trivially_move_constructible_v<default_values_Store>);
    static_assert(std::is_trivially_copy_assignable_v<default_values_Store>);
    static_assert(std::is_trivially_move_assignable_v<default_values_Store>);
    static_assert(std::is_trivially_destructible_v<default_values_Store>);
    static default_values_Store default_values_global;
    auto X0_default_values() -> std::decay<decltype(default_values_global.X0)>::type  {
        return default_values_global.X0;
    }
    auto Z0_default_values() -> std::decay<decltype(default_values_global.Z0)>::type  {
        return default_values_global.Z0;
    }
    auto A0_default_values() -> std::decay<decltype(default_values_global.A0)>::type  {
        return default_values_global.A0;
    }
    auto B0_default_values() -> std::decay<decltype(default_values_global.B0)>::type  {
        return default_values_global.B0;
    }
    auto Y0_default_values() -> std::decay<decltype(default_values_global.Y0)>::type  {
        return default_values_global.Y0;
    }

    static std::vector<double> _parameter_defaults = {
    };


    /** all mechanism instance variables and global variables */
    struct default_values_Instance  {
        double* X{};
        double* Y{};
        double* Z{};
        double* A{};
        double* B{};
        double* DX{};
        double* DY{};
        double* DZ{};
        double* DA{};
        double* DB{};
        double* v_unused{};
        default_values_Store* global{&default_values_global};
    };


    struct default_values_NodeData  {
        int const * nodeindices;
        double const * node_voltages;
        double * node_diagonal;
        double * node_rhs;
        int nodecount;
    };


    static default_values_Instance make_instance_default_values(_nrn_mechanism_cache_range* _lmc) {
        if(_lmc == nullptr) {
            return default_values_Instance();
        }

        return default_values_Instance {
            _lmc->template fpfield_ptr<0>(),
            _lmc->template fpfield_ptr<1>(),
            _lmc->template fpfield_ptr<2>(),
            _lmc->template data_array_ptr<3, 3>(),
            _lmc->template data_array_ptr<4, 2>(),
            _lmc->template fpfield_ptr<5>(),
            _lmc->template fpfield_ptr<6>(),
            _lmc->template fpfield_ptr<7>(),
            _lmc->template data_array_ptr<8, 3>(),
            _lmc->template data_array_ptr<9, 2>(),
            _lmc->template fpfield_ptr<10>()
        };
    }


    static default_values_NodeData make_node_data_default_values(NrnThread& nt, Memb_list& _ml_arg) {
        return default_values_NodeData {
            _ml_arg.nodeindices,
            nt.node_voltage_storage(),
            nt.node_d_storage(),
            nt.node_rhs_storage(),
            _ml_arg.nodecount
        };
    }
    static default_values_NodeData make_node_data_default_values(Prop * _prop) {
        if(!_prop) {
            return default_values_NodeData();
        }

        static std::vector<int> node_index{0};
        Node* _node = _nrn_mechanism_access_node(_prop);
        return default_values_NodeData {
            node_index.data(),
            &_nrn_mechanism_access_voltage(_node),
            &_nrn_mechanism_access_d(_node),
            &_nrn_mechanism_access_rhs(_node),
            1
        };
    }

    static void nrn_destructor_default_values(Prop* prop);


    static void nrn_alloc_default_values(Prop* _prop) {
        Datum *_ppvar = nullptr;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const _iml = 0;
        assert(_nrn_mechanism_get_num_vars(_prop) == 11);
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
        {"X0_default_values", &default_values_global.X0},
        {"Z0_default_values", &default_values_global.Z0},
        {"A0_default_values", &default_values_global.A0},
        {"B0_default_values", &default_values_global.B0},
        {nullptr, nullptr}
    };


    /** connect global (array) variables to hoc -- */
    static DoubVec hoc_vector_double[] = {
        {nullptr, nullptr, 0}
    };


    /* declaration of user functions */


    /* connect user functions to hoc names */
    static VoidFunc hoc_intfunc[] = {
        {"setdata_default_values", _hoc_setdata},
        {nullptr, nullptr}
    };
    static NPyDirectMechFunc npy_direct_func_proc[] = {
        {nullptr, nullptr}
    };


    static void nrn_init_default_values(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_default_values(&_lmc);
        auto node_data = make_node_data_default_values(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
            auto* _ppvar = _ml_arg->pdata[id];
            int node_id = node_data.nodeindices[id];
            inst.v_unused[id] = node_data.node_voltages[node_id];
            inst.X[id] = inst.global->X0;
            inst.Y[id] = inst.global->Y0;
            inst.Z[id] = inst.global->Z0;
            (inst.A+id*3)[0] = inst.global->A0;
            (inst.A+id*3)[1] = inst.global->A0;
            (inst.A+id*3)[2] = inst.global->A0;
            (inst.B+id*2)[0] = inst.global->B0;
            (inst.B+id*2)[1] = inst.global->B0;
            inst.Z[id] = 7.0;
            (inst.B+id*2)[static_cast<int>(1)] = 8.0;
        }
    }


    static void nrn_jacob_default_values(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_default_values(&_lmc);
        auto node_data = make_node_data_default_values(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
        }
    }
    static void nrn_destructor_default_values(Prop* prop) {
        Datum* _ppvar = _nrn_mechanism_access_dparam(prop);
        _nrn_mechanism_cache_instance _lmc{prop};
        const size_t id = 0;
        auto inst = make_instance_default_values(prop ? &_lmc : nullptr);
        auto node_data = make_node_data_default_values(prop);

    }


    static void _initlists() {
    }


    extern "C" void _default_values_reg() {
        _initlists();

        register_mech(mechanism_info, nrn_alloc_default_values, nullptr, nullptr, nullptr, nrn_init_default_values, -1, 1);

        mech_type = nrn_get_mechtype(mechanism_info[1]);
        hoc_register_parm_default(mech_type, &_parameter_defaults);
        _nrn_mechanism_register_data_fields(mech_type,
            _nrn_mechanism_field<double>{"X"} /* 0 */,
            _nrn_mechanism_field<double>{"Y"} /* 1 */,
            _nrn_mechanism_field<double>{"Z"} /* 2 */,
            _nrn_mechanism_field<double>{"A", 3} /* 3 */,
            _nrn_mechanism_field<double>{"B", 2} /* 4 */,
            _nrn_mechanism_field<double>{"DX"} /* 5 */,
            _nrn_mechanism_field<double>{"DY"} /* 6 */,
            _nrn_mechanism_field<double>{"DZ"} /* 7 */,
            _nrn_mechanism_field<double>{"DA", 3} /* 8 */,
            _nrn_mechanism_field<double>{"DB", 2} /* 9 */,
            _nrn_mechanism_field<double>{"v_unused"} /* 10 */
        );

        hoc_register_prop_size(mech_type, 17, 0);
        hoc_register_var(hoc_scalar_double, hoc_vector_double, hoc_intfunc);
        hoc_register_npy_direct(mech_type, npy_direct_func_proc);
    }
}
