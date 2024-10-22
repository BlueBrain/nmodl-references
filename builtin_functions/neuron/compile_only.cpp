/*********************************************************
Model Name      : compile_only
Filename        : compile_only.mod
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
        "compile_only",
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
    struct compile_only_Store {
    };
    static_assert(std::is_trivially_copy_constructible_v<compile_only_Store>);
    static_assert(std::is_trivially_move_constructible_v<compile_only_Store>);
    static_assert(std::is_trivially_copy_assignable_v<compile_only_Store>);
    static_assert(std::is_trivially_move_assignable_v<compile_only_Store>);
    static_assert(std::is_trivially_destructible_v<compile_only_Store>);
    static compile_only_Store compile_only_global;
    static std::vector<double> _parameter_defaults = {
    };


    /** all mechanism instance variables and global variables */
    struct compile_only_Instance  {
        double* v_unused{};
        compile_only_Store* global{&compile_only_global};
    };


    struct compile_only_NodeData  {
        int const * nodeindices;
        double const * node_voltages;
        double * node_diagonal;
        double * node_rhs;
        int nodecount;
    };


    static compile_only_Instance make_instance_compile_only(_nrn_mechanism_cache_range* _lmc) {
        if(_lmc == nullptr) {
            return compile_only_Instance();
        }

        return compile_only_Instance {
            _lmc->template fpfield_ptr<0>()
        };
    }


    static compile_only_NodeData make_node_data_compile_only(NrnThread& nt, Memb_list& _ml_arg) {
        return compile_only_NodeData {
            _ml_arg.nodeindices,
            nt.node_voltage_storage(),
            nt.node_d_storage(),
            nt.node_rhs_storage(),
            _ml_arg.nodecount
        };
    }
    static compile_only_NodeData make_node_data_compile_only(Prop * _prop) {
        if(!_prop) {
            return compile_only_NodeData();
        }

        static std::vector<int> node_index{0};
        Node* _node = _nrn_mechanism_access_node(_prop);
        return compile_only_NodeData {
            node_index.data(),
            &_nrn_mechanism_access_voltage(_node),
            &_nrn_mechanism_access_d(_node),
            &_nrn_mechanism_access_rhs(_node),
            1
        };
    }

    static void nrn_destructor_compile_only(Prop* prop);


    static void nrn_alloc_compile_only(Prop* _prop) {
        Datum *_ppvar = nullptr;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const _iml = 0;
        assert(_nrn_mechanism_get_num_vars(_prop) == 1);
        /*initialize range parameters*/
    }


    /* Mechanism procedures and functions */
    inline static double call_nrn_ghk_compile_only(_nrn_mechanism_cache_range& _lmc, compile_only_Instance& inst, compile_only_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt);
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
    static void _hoc_call_nrn_ghk();
    static double _npy_call_nrn_ghk(Prop* _prop);


    /* connect user functions to hoc names */
    static VoidFunc hoc_intfunc[] = {
        {"setdata_compile_only", _hoc_setdata},
        {"call_nrn_ghk_compile_only", _hoc_call_nrn_ghk},
        {nullptr, nullptr}
    };
    static NPyDirectMechFunc npy_direct_func_proc[] = {
        {"call_nrn_ghk", _npy_call_nrn_ghk},
        {nullptr, nullptr}
    };
    static void _hoc_call_nrn_ghk() {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        Prop* _local_prop = _prop_id ? _extcall_prop : nullptr;
        _nrn_mechanism_cache_instance _lmc{_local_prop};
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_compile_only(_local_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_compile_only(_local_prop);
        double _r = 0.0;
        _r = call_nrn_ghk_compile_only(_lmc, inst, node_data, id, _ppvar, _thread, nt);
        hoc_retpushx(_r);
    }
    static double _npy_call_nrn_ghk(Prop* _prop) {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id = 0;
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_compile_only(_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_compile_only(_prop);
        double _r = 0.0;
        _r = call_nrn_ghk_compile_only(_lmc, inst, node_data, id, _ppvar, _thread, nt);
        return(_r);
    }


    inline double call_nrn_ghk_compile_only(_nrn_mechanism_cache_range& _lmc, compile_only_Instance& inst, compile_only_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt) {
        double ret_call_nrn_ghk = 0.0;
        double v = node_data.node_voltages ? node_data.node_voltages[node_data.nodeindices[id]] : 0.0;
        ret_call_nrn_ghk = nrn_ghk(1.0, 2.0, 3.0, 4.0);
        return ret_call_nrn_ghk;
    }


    static void nrn_init_compile_only(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_compile_only(&_lmc);
        auto node_data = make_node_data_compile_only(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
            auto* _ppvar = _ml_arg->pdata[id];
            int node_id = node_data.nodeindices[id];
            auto v = node_data.node_voltages[node_id];
        }
    }


    static void nrn_jacob_compile_only(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_compile_only(&_lmc);
        auto node_data = make_node_data_compile_only(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
        }
    }
    static void nrn_destructor_compile_only(Prop* prop) {
        Datum* _ppvar = _nrn_mechanism_access_dparam(prop);
        _nrn_mechanism_cache_instance _lmc{prop};
        const size_t id = 0;
        auto inst = make_instance_compile_only(prop ? &_lmc : nullptr);
        auto node_data = make_node_data_compile_only(prop);

    }


    static void _initlists() {
    }


    /** register channel with the simulator */
    extern "C" void _compile_only_reg() {
        _initlists();

        register_mech(mechanism_info, nrn_alloc_compile_only, nullptr, nullptr, nullptr, nrn_init_compile_only, -1, 1);

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
