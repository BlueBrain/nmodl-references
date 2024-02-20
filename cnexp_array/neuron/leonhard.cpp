/*********************************************************
Model Name      : leonhard
Filename        : leonhard.mod
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
        "leonhard",
        0,
        "z_leonhard[3]",
        0,
        "x_leonhard",
        "s_leonhard[2]",
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
    struct leonhard_Store {
        double x0{};
        double s0{};
    };
    static_assert(std::is_trivially_copy_constructible_v<leonhard_Store>);
    static_assert(std::is_trivially_move_constructible_v<leonhard_Store>);
    static_assert(std::is_trivially_copy_assignable_v<leonhard_Store>);
    static_assert(std::is_trivially_move_assignable_v<leonhard_Store>);
    static_assert(std::is_trivially_destructible_v<leonhard_Store>);
    leonhard_Store leonhard_global;


    /** all mechanism instance variables and global variables */
    struct leonhard_Instance  {
        double* z{};
        double* x{};
        double* s{};
        double* Dx{};
        double* Ds{};
        double* v_unused{};
        double* g_unused{};
        leonhard_Store* global{&leonhard_global};
    };


    static leonhard_Instance make_instance_leonhard(_nrn_mechanism_cache_range& _ml) {
        return leonhard_Instance {
            _ml.template data_array_ptr<0, 3>(),
            _ml.template fpfield_ptr<1>(),
            _ml.template data_array_ptr<2, 2>(),
            _ml.template fpfield_ptr<3>(),
            _ml.template data_array_ptr<4, 2>(),
            _ml.template fpfield_ptr<5>(),
            _ml.template fpfield_ptr<6>()
        };
    }


    static void nrn_alloc_leonhard(Prop* _prop) {
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
        {"setdata_leonhard", _hoc_setdata},
        {0, 0}
    };
    static NPyDirectMechFunc npy_direct_func_proc[] = {
    };


    void nrn_init_leonhard(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
        auto inst = make_instance_leonhard(_lmr);
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
            inst.x[id] = 42.0;
            (inst.s+id*2)[static_cast<int>(0)] = 0.1;
            (inst.s+id*2)[static_cast<int>(1)] =  -1.0;
            (inst.z+id*3)[static_cast<int>(0)] = 0.7;
            (inst.z+id*3)[static_cast<int>(1)] = 0.8;
            (inst.z+id*3)[static_cast<int>(2)] = 0.9;
        }
    }


    void nrn_state_leonhard(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
        auto inst = make_instance_leonhard(_lmr);
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
            inst.x[id] = inst.x[id] + (1.0 - exp(_nt->_dt * ((((inst.s+id*2)[static_cast<int>(0)] + (inst.s+id*2)[static_cast<int>(1)]) * ((inst.z+id*3)[static_cast<int>(0)] * (inst.z+id*3)[static_cast<int>(1)] * (inst.z+id*3)[static_cast<int>(2)])) * (1.0)))) * ( -(0.0) / (((((inst.s+id*2)[static_cast<int>(0)] + (inst.s+id*2)[static_cast<int>(1)])) * (((((inst.z+id*3)[static_cast<int>(0)]) * ((inst.z+id*3)[static_cast<int>(1)])) * ((inst.z+id*3)[static_cast<int>(2)])))) * (1.0)) - inst.x[id]);
        }
    }


    /** nrn_jacob function */
    static void nrn_jacob_leonhard(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {}


    static void _initlists() {
        /* x */
        _slist1[0] = {1, 0};
        /* Dx */
        _dlist1[0] = {3, 0};
    }


    /** register channel with the simulator */
    extern "C" void _leonhard_reg() {
        _initlists();



        register_mech(mechanism_info, nrn_alloc_leonhard, nullptr, nrn_jacob_leonhard, nrn_state_leonhard, nrn_init_leonhard, hoc_nrnpointerindex, 1);

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

        hoc_register_prop_size(mech_type, 7, 0);
        hoc_register_var(hoc_scalar_double, hoc_vector_double, hoc_intfunc);
        hoc_register_npy_direct(mech_type, npy_direct_func_proc);
    }
}
