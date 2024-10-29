/*********************************************************
Model Name      : random_variable
Filename        : random_variable.mod
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

static constexpr auto number_of_datum_variables = 1;
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
        "random_variable",
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
    struct random_variable_Store {
    };
    static_assert(std::is_trivially_copy_constructible_v<random_variable_Store>);
    static_assert(std::is_trivially_move_constructible_v<random_variable_Store>);
    static_assert(std::is_trivially_copy_assignable_v<random_variable_Store>);
    static_assert(std::is_trivially_move_assignable_v<random_variable_Store>);
    static_assert(std::is_trivially_destructible_v<random_variable_Store>);
    static random_variable_Store random_variable_global;
    static std::vector<double> _parameter_defaults = {
    };


    /** all mechanism instance variables and global variables */
    struct random_variable_Instance  {
        double* v_unused{};
        void** const* rng{};
        random_variable_Store* global{&random_variable_global};
    };


    struct random_variable_NodeData  {
        int const * nodeindices;
        double const * node_voltages;
        double * node_diagonal;
        double * node_rhs;
        int nodecount;
    };


    static random_variable_Instance make_instance_random_variable(_nrn_mechanism_cache_range* _lmc) {
        if(_lmc == nullptr) {
            return random_variable_Instance();
        }

        return random_variable_Instance {
            _lmc->template fpfield_ptr<0>()
        };
    }


    static random_variable_NodeData make_node_data_random_variable(NrnThread& nt, Memb_list& _ml_arg) {
        return random_variable_NodeData {
            _ml_arg.nodeindices,
            nt.node_voltage_storage(),
            nt.node_d_storage(),
            nt.node_rhs_storage(),
            _ml_arg.nodecount
        };
    }
    static random_variable_NodeData make_node_data_random_variable(Prop * _prop) {
        if(!_prop) {
            return random_variable_NodeData();
        }

        static std::vector<int> node_index{0};
        Node* _node = _nrn_mechanism_access_node(_prop);
        return random_variable_NodeData {
            node_index.data(),
            &_nrn_mechanism_access_voltage(_node),
            &_nrn_mechanism_access_d(_node),
            &_nrn_mechanism_access_rhs(_node),
            1
        };
    }

    static void nrn_destructor_random_variable(Prop* prop);


    static void nrn_alloc_random_variable(Prop* _prop) {
        Datum *_ppvar = nullptr;
        _ppvar = nrn_prop_datum_alloc(mech_type, 1, _prop);
        _nrn_mechanism_access_dparam(_prop) = _ppvar;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const _iml = 0;
        assert(_nrn_mechanism_get_num_vars(_prop) == 1);
        /*initialize range parameters*/
        _nrn_mechanism_access_dparam(_prop) = _ppvar;
        _ppvar[0].literal_value<void*>() = nrnran123_newstream();
        nrn_mech_inst_destruct[mech_type] = neuron::nrn_destructor_random_variable;
    }


    /* Mechanism procedures and functions */
    inline static double negexp_random_variable(_nrn_mechanism_cache_range& _lmc, random_variable_Instance& inst, random_variable_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt);
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
        {nullptr, nullptr}
    };


    /** connect global (array) variables to hoc -- */
    static DoubVec hoc_vector_double[] = {
        {nullptr, nullptr, 0}
    };


    /* declaration of user functions */
    static void _hoc_negexp();
    static double _npy_negexp(Prop* _prop);


    /* connect user functions to hoc names */
    static VoidFunc hoc_intfunc[] = {
        {"setdata_random_variable", _hoc_setdata},
        {"negexp_random_variable", _hoc_negexp},
        {nullptr, nullptr}
    };
    static NPyDirectMechFunc npy_direct_func_proc[] = {
        {"negexp", _npy_negexp},
        {nullptr, nullptr}
    };
    static void _hoc_negexp() {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        Prop* _local_prop = _prop_id ? _extcall_prop : nullptr;
        _nrn_mechanism_cache_instance _lmc{_local_prop};
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_random_variable(_local_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_random_variable(_local_prop);
        double _r = 0.0;
        _r = negexp_random_variable(_lmc, inst, node_data, id, _ppvar, _thread, nt);
        hoc_retpushx(_r);
    }
    static double _npy_negexp(Prop* _prop) {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id = 0;
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_random_variable(_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_random_variable(_prop);
        double _r = 0.0;
        _r = negexp_random_variable(_lmc, inst, node_data, id, _ppvar, _thread, nt);
        return(_r);
    }


    inline double negexp_random_variable(_nrn_mechanism_cache_range& _lmc, random_variable_Instance& inst, random_variable_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt) {
        double ret_negexp = 0.0;
        double v = node_data.node_voltages ? node_data.node_voltages[node_data.nodeindices[id]] : 0.0;
        ret_negexp = nrnran123_negexp((nrnran123_State*)_ppvar[0].literal_value<void*>());
        return ret_negexp;
    }


    static void nrn_init_random_variable(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_random_variable(&_lmc);
        auto node_data = make_node_data_random_variable(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        #pragma omp simd
        #pragma ivdep
        for (int id = 0; id < nodecount; id++) {
            auto* _ppvar = _ml_arg->pdata[id];
            int node_id = node_data.nodeindices[id];
            inst.v_unused[id] = node_data.node_voltages[node_id];
        }
    }


    static void nrn_jacob_random_variable(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_random_variable(&_lmc);
        auto node_data = make_node_data_random_variable(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        #pragma omp simd
        #pragma ivdep
        for (int id = 0; id < nodecount; id++) {
        }
    }
    static void nrn_destructor_random_variable(Prop* prop) {
        Datum* _ppvar = _nrn_mechanism_access_dparam(prop);
        _nrn_mechanism_cache_instance _lmc{prop};
        const size_t id = 0;
        auto inst = make_instance_random_variable(prop ? &_lmc : nullptr);
        auto node_data = make_node_data_random_variable(prop);

        nrnran123_deletestream((nrnran123_State*) _ppvar[0].literal_value<void*>());
    }


    static void _initlists() {
    }


    extern "C" void _random_variable_reg() {
        _initlists();

        register_mech(mechanism_info, nrn_alloc_random_variable, nullptr, nullptr, nullptr, nrn_init_random_variable, -1, 1);

        mech_type = nrn_get_mechtype(mechanism_info[1]);
        hoc_register_parm_default(mech_type, &_parameter_defaults);
        _nrn_mechanism_register_data_fields(mech_type,
            _nrn_mechanism_field<double>{"v_unused"} /* 0 */,
            _nrn_mechanism_field<double*>{"rng", "random"} /* 0 */
        );

        hoc_register_prop_size(mech_type, 1, 1);
        hoc_register_dparam_semantics(mech_type, 0, "random");
        hoc_register_var(hoc_scalar_double, hoc_vector_double, hoc_intfunc);
        hoc_register_npy_direct(mech_type, npy_direct_func_proc);
    }
}
