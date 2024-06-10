/*********************************************************
Model Name      : cnexp_array
Filename        : cnexp_array.mod
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
static constexpr auto number_of_floating_point_variables = 7;

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
        "cnexp_array",
        0,
        "z_cnexp_array[3]",
        0,
        "x_cnexp_array",
        "s_cnexp_array[2]",
        0,
        0
    };


    /* NEURON global variables */
    static neuron::container::field_index _slist1[1], _dlist1[1];
    static int mech_type;
    static Prop* _extcall_prop;
    /* _prop_id kind of shadows _extcall_prop to allow validity checking. */
    static _nrn_non_owning_id_without_container _prop_id{};
    static int hoc_nrnpointerindex = -1;
    static _nrn_mechanism_std_vector<Datum> _extcall_thread;


    /** all global variables */
    struct cnexp_array_Store {
        double x0{};
        double s0{};
    };
    static_assert(std::is_trivially_copy_constructible_v<cnexp_array_Store>);
    static_assert(std::is_trivially_move_constructible_v<cnexp_array_Store>);
    static_assert(std::is_trivially_copy_assignable_v<cnexp_array_Store>);
    static_assert(std::is_trivially_move_assignable_v<cnexp_array_Store>);
    static_assert(std::is_trivially_destructible_v<cnexp_array_Store>);
    cnexp_array_Store cnexp_array_global;


    /** all mechanism instance variables and global variables */
    struct cnexp_array_Instance  {
        double* z{};
        double* x{};
        double* s{};
        double* Dx{};
        double* Ds{};
        double* v_unused{};
        double* g_unused{};
        cnexp_array_Store* global{&cnexp_array_global};
    };


    struct cnexp_array_NodeData  {
        int const * nodeindices;
        double const * node_voltages;
        double * node_diagonal;
        double * node_rhs;
        int nodecount;
    };


    static cnexp_array_Instance make_instance_cnexp_array(_nrn_mechanism_cache_range& _ml) {
        return cnexp_array_Instance {
            _ml.template data_array_ptr<0, 3>(),
            _ml.template fpfield_ptr<1>(),
            _ml.template data_array_ptr<2, 2>(),
            _ml.template fpfield_ptr<3>(),
            _ml.template data_array_ptr<4, 2>(),
            _ml.template fpfield_ptr<5>(),
            _ml.template fpfield_ptr<6>()
        };
    }


    static cnexp_array_NodeData make_node_data_cnexp_array(NrnThread& _nt, Memb_list& _ml_arg) {
        return cnexp_array_NodeData {
            _ml_arg.nodeindices,
            _nt.node_voltage_storage(),
            _nt.node_d_storage(),
            _nt.node_rhs_storage(),
            _ml_arg.nodecount
        };
    }


    static void nrn_alloc_cnexp_array(Prop* _prop) {
        Prop *prop_ion{};
        Datum *_ppvar{};
        _nrn_mechanism_cache_instance _ml_real{_prop};
        auto* const _ml = &_ml_real;
        size_t const _iml{};
        assert(_nrn_mechanism_get_num_vars(_prop) == 7);
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
        {"setdata_cnexp_array", _hoc_setdata},
        {nullptr, nullptr}
    };
    static NPyDirectMechFunc npy_direct_func_proc[] = {
        {nullptr, nullptr}
    };


    void nrn_init_cnexp_array(const _nrn_model_sorted_token& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
        auto inst = make_instance_cnexp_array(_lmr);
        auto node_data = make_node_data_cnexp_array(*_nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* const _ml = &_lmr;
        auto* _thread = _ml_arg->_thread;
        for (int id = 0; id < nodecount; id++) {
            auto* _ppvar = _ml_arg->pdata[id];
            int node_id = node_data.nodeindices[id];
            auto v = node_data.node_voltages[node_id];
            inst.v_unused[id] = v;
            inst.x[id] = 42.0;
            (inst.s+id*2)[static_cast<int>(0)] = 0.1;
            (inst.s+id*2)[static_cast<int>(1)] =  -1.0;
            (inst.z+id*3)[static_cast<int>(0)] = 0.7;
            (inst.z+id*3)[static_cast<int>(1)] = 0.8;
            (inst.z+id*3)[static_cast<int>(2)] = 0.9;
        }
    }


    void nrn_state_cnexp_array(const _nrn_model_sorted_token& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
        auto inst = make_instance_cnexp_array(_lmr);
        auto node_data = make_node_data_cnexp_array(*_nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* const _ml = &_lmr;
        auto* _thread = _ml_arg->_thread;
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_data.nodeindices[id];
            auto* _ppvar = _ml_arg->pdata[id];
            auto v = node_data.node_voltages[node_id];
            inst.x[id] = inst.x[id] + (1.0 - exp(_nt->_dt * ((((inst.s+id*2)[static_cast<int>(0)] + (inst.s+id*2)[static_cast<int>(1)]) * ((inst.z+id*3)[static_cast<int>(0)] * (inst.z+id*3)[static_cast<int>(1)] * (inst.z+id*3)[static_cast<int>(2)])) * (1.0)))) * ( -(0.0) / (((((inst.s+id*2)[static_cast<int>(0)] + (inst.s+id*2)[static_cast<int>(1)])) * (((((inst.z+id*3)[static_cast<int>(0)]) * ((inst.z+id*3)[static_cast<int>(1)])) * ((inst.z+id*3)[static_cast<int>(2)])))) * (1.0)) - inst.x[id]);
        }
    }


    static void nrn_jacob_cnexp_array(const _nrn_model_sorted_token& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
        auto inst = make_instance_cnexp_array(_lmr);
        auto node_data = make_node_data_cnexp_array(*_nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_data.nodeindices[id];
            node_data.node_diagonal[node_id] += inst.g_unused[id];
        }
    }


    static void _initlists() {
        /* x */
        _slist1[0] = {1, 0};
        /* Dx */
        _dlist1[0] = {3, 0};
    }


    /** register channel with the simulator */
    extern "C" void _cnexp_array_reg() {
        _initlists();

        register_mech(mechanism_info, nrn_alloc_cnexp_array, nullptr, nrn_jacob_cnexp_array, nrn_state_cnexp_array, nrn_init_cnexp_array, hoc_nrnpointerindex, 1);

        mech_type = nrn_get_mechtype(mechanism_info[1]);
        _nrn_mechanism_register_data_fields(mech_type,
            _nrn_mechanism_field<double>{"z", 3} /* 0 */,
            _nrn_mechanism_field<double>{"x"} /* 1 */,
            _nrn_mechanism_field<double>{"s", 2} /* 2 */,
            _nrn_mechanism_field<double>{"Dx"} /* 3 */,
            _nrn_mechanism_field<double>{"Ds", 2} /* 4 */,
            _nrn_mechanism_field<double>{"v_unused"} /* 5 */,
            _nrn_mechanism_field<double>{"g_unused"} /* 6 */
        );

        hoc_register_prop_size(mech_type, 11, 0);
        hoc_register_var(hoc_scalar_double, hoc_vector_double, hoc_intfunc);
        hoc_register_npy_direct(mech_type, npy_direct_func_proc);
    }
}
