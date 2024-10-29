/*********************************************************
Model Name      : functions
Filename        : functions.mod
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
static constexpr auto number_of_floating_point_variables = 2;

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
        "functions",
        0,
        "x_functions",
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
    struct functions_Store {
    };
    static_assert(std::is_trivially_copy_constructible_v<functions_Store>);
    static_assert(std::is_trivially_move_constructible_v<functions_Store>);
    static_assert(std::is_trivially_copy_assignable_v<functions_Store>);
    static_assert(std::is_trivially_move_assignable_v<functions_Store>);
    static_assert(std::is_trivially_destructible_v<functions_Store>);
    static functions_Store functions_global;
    static std::vector<double> _parameter_defaults = {
    };


    /** all mechanism instance variables and global variables */
    struct functions_Instance  {
        double* x{};
        double* v_unused{};
        functions_Store* global{&functions_global};
    };


    struct functions_NodeData  {
        int const * nodeindices;
        double const * node_voltages;
        double * node_diagonal;
        double * node_rhs;
        int nodecount;
    };


    static functions_Instance make_instance_functions(_nrn_mechanism_cache_range* _lmc) {
        if(_lmc == nullptr) {
            return functions_Instance();
        }

        return functions_Instance {
            _lmc->template fpfield_ptr<0>(),
            _lmc->template fpfield_ptr<1>()
        };
    }


    static functions_NodeData make_node_data_functions(NrnThread& nt, Memb_list& _ml_arg) {
        return functions_NodeData {
            _ml_arg.nodeindices,
            nt.node_voltage_storage(),
            nt.node_d_storage(),
            nt.node_rhs_storage(),
            _ml_arg.nodecount
        };
    }
    static functions_NodeData make_node_data_functions(Prop * _prop) {
        if(!_prop) {
            return functions_NodeData();
        }

        static std::vector<int> node_index{0};
        Node* _node = _nrn_mechanism_access_node(_prop);
        return functions_NodeData {
            node_index.data(),
            &_nrn_mechanism_access_voltage(_node),
            &_nrn_mechanism_access_d(_node),
            &_nrn_mechanism_access_rhs(_node),
            1
        };
    }

    static void nrn_destructor_functions(Prop* prop);


    static void nrn_alloc_functions(Prop* _prop) {
        Datum *_ppvar = nullptr;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const _iml = 0;
        assert(_nrn_mechanism_get_num_vars(_prop) == 2);
        /*initialize range parameters*/
    }


    /* Mechanism procedures and functions */
    inline static double x_plus_a_functions(_nrn_mechanism_cache_range& _lmc, functions_Instance& inst, functions_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _la);
    inline static double v_plus_a_functions(_nrn_mechanism_cache_range& _lmc, functions_Instance& inst, functions_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _la);
    inline static double identity_functions(_nrn_mechanism_cache_range& _lmc, functions_Instance& inst, functions_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _lv);
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
    static void _hoc_x_plus_a();
    static double _npy_x_plus_a(Prop* _prop);
    static void _hoc_v_plus_a();
    static double _npy_v_plus_a(Prop* _prop);
    static void _hoc_identity();
    static double _npy_identity(Prop* _prop);


    /* connect user functions to hoc names */
    static VoidFunc hoc_intfunc[] = {
        {"setdata_functions", _hoc_setdata},
        {"x_plus_a_functions", _hoc_x_plus_a},
        {"v_plus_a_functions", _hoc_v_plus_a},
        {"identity_functions", _hoc_identity},
        {nullptr, nullptr}
    };
    static NPyDirectMechFunc npy_direct_func_proc[] = {
        {"x_plus_a", _npy_x_plus_a},
        {"v_plus_a", _npy_v_plus_a},
        {"identity", _npy_identity},
        {nullptr, nullptr}
    };
    static void _hoc_x_plus_a() {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        if (!_prop_id) {
            hoc_execerror("No data for x_plus_a_functions. Requires prior call to setdata_functions and that the specified mechanism instance still be in existence.", nullptr);
        }
        Prop* _local_prop = _extcall_prop;
        _nrn_mechanism_cache_instance _lmc{_local_prop};
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_functions(_local_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_functions(_local_prop);
        double _r = 0.0;
        _r = x_plus_a_functions(_lmc, inst, node_data, id, _ppvar, _thread, nt, *getarg(1));
        hoc_retpushx(_r);
    }
    static double _npy_x_plus_a(Prop* _prop) {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id = 0;
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_functions(_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_functions(_prop);
        double _r = 0.0;
        _r = x_plus_a_functions(_lmc, inst, node_data, id, _ppvar, _thread, nt, *getarg(1));
        return(_r);
    }
    static void _hoc_v_plus_a() {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        Prop* _local_prop = _prop_id ? _extcall_prop : nullptr;
        _nrn_mechanism_cache_instance _lmc{_local_prop};
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_functions(_local_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_functions(_local_prop);
        double _r = 0.0;
        _r = v_plus_a_functions(_lmc, inst, node_data, id, _ppvar, _thread, nt, *getarg(1));
        hoc_retpushx(_r);
    }
    static double _npy_v_plus_a(Prop* _prop) {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id = 0;
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_functions(_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_functions(_prop);
        double _r = 0.0;
        _r = v_plus_a_functions(_lmc, inst, node_data, id, _ppvar, _thread, nt, *getarg(1));
        return(_r);
    }
    static void _hoc_identity() {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        Prop* _local_prop = _prop_id ? _extcall_prop : nullptr;
        _nrn_mechanism_cache_instance _lmc{_local_prop};
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_functions(_local_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_functions(_local_prop);
        double _r = 0.0;
        _r = identity_functions(_lmc, inst, node_data, id, _ppvar, _thread, nt, *getarg(1));
        hoc_retpushx(_r);
    }
    static double _npy_identity(Prop* _prop) {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id = 0;
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_functions(_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_functions(_prop);
        double _r = 0.0;
        _r = identity_functions(_lmc, inst, node_data, id, _ppvar, _thread, nt, *getarg(1));
        return(_r);
    }


    inline double x_plus_a_functions(_nrn_mechanism_cache_range& _lmc, functions_Instance& inst, functions_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _la) {
        double ret_x_plus_a = 0.0;
        double v = node_data.node_voltages ? node_data.node_voltages[node_data.nodeindices[id]] : 0.0;
        ret_x_plus_a = inst.x[id] + _la;
        return ret_x_plus_a;
    }


    inline double v_plus_a_functions(_nrn_mechanism_cache_range& _lmc, functions_Instance& inst, functions_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _la) {
        double ret_v_plus_a = 0.0;
        double v = node_data.node_voltages ? node_data.node_voltages[node_data.nodeindices[id]] : 0.0;
        ret_v_plus_a = inst.v_unused[id] + _la;
        return ret_v_plus_a;
    }


    inline double identity_functions(_nrn_mechanism_cache_range& _lmc, functions_Instance& inst, functions_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _lv) {
        double ret_identity = 0.0;
        double v = node_data.node_voltages ? node_data.node_voltages[node_data.nodeindices[id]] : 0.0;
        ret_identity = _lv;
        return ret_identity;
    }


    static void nrn_init_functions(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_functions(&_lmc);
        auto node_data = make_node_data_functions(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        #pragma omp simd
        #pragma ivdep
        for (int id = 0; id < nodecount; id++) {
            auto* _ppvar = _ml_arg->pdata[id];
            int node_id = node_data.nodeindices[id];
            inst.v_unused[id] = node_data.node_voltages[node_id];
            inst.x[id] = 1.0;
        }
    }


    static void nrn_jacob_functions(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_functions(&_lmc);
        auto node_data = make_node_data_functions(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        #pragma omp simd
        #pragma ivdep
        for (int id = 0; id < nodecount; id++) {
        }
    }
    static void nrn_destructor_functions(Prop* prop) {
        Datum* _ppvar = _nrn_mechanism_access_dparam(prop);
        _nrn_mechanism_cache_instance _lmc{prop};
        const size_t id = 0;
        auto inst = make_instance_functions(prop ? &_lmc : nullptr);
        auto node_data = make_node_data_functions(prop);

    }


    static void _initlists() {
    }


    extern "C" void _functions_reg() {
        _initlists();

        register_mech(mechanism_info, nrn_alloc_functions, nullptr, nullptr, nullptr, nrn_init_functions, -1, 1);

        mech_type = nrn_get_mechtype(mechanism_info[1]);
        hoc_register_parm_default(mech_type, &_parameter_defaults);
        _nrn_mechanism_register_data_fields(mech_type,
            _nrn_mechanism_field<double>{"x"} /* 0 */,
            _nrn_mechanism_field<double>{"v_unused"} /* 1 */
        );

        hoc_register_prop_size(mech_type, 2, 0);
        hoc_register_var(hoc_scalar_double, hoc_vector_double, hoc_intfunc);
        hoc_register_npy_direct(mech_type, npy_direct_func_proc);
    }
}
