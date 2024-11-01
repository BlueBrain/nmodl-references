/*********************************************************
Model Name      : top_local
Filename        : top_local.mod
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
extern void _nrn_thread_reg(int, int, void(*)(Datum*));

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
        "top_local",
        0,
        "y_top_local",
        "il_top_local",
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
    struct top_local_Store {
        double thread_data_in_use{0};
        double thread_data[1] /* TODO init const-array */;
    };
    static_assert(std::is_trivially_copy_constructible_v<top_local_Store>);
    static_assert(std::is_trivially_move_constructible_v<top_local_Store>);
    static_assert(std::is_trivially_copy_assignable_v<top_local_Store>);
    static_assert(std::is_trivially_move_assignable_v<top_local_Store>);
    static_assert(std::is_trivially_destructible_v<top_local_Store>);
    static top_local_Store top_local_global;
    auto thread_data_in_use_top_local() -> std::decay<decltype(top_local_global.thread_data_in_use)>::type  {
        return top_local_global.thread_data_in_use;
    }
    auto thread_data_top_local() -> std::decay<decltype(top_local_global.thread_data)>::type  {
        return top_local_global.thread_data;
    }

    static std::vector<double> _parameter_defaults = {
    };


    /** all mechanism instance variables and global variables */
    struct top_local_Instance  {
        double* y{};
        double* il{};
        double* v_unused{};
        double* g_unused{};
        top_local_Store* global{&top_local_global};
    };


    struct top_local_NodeData  {
        int const * nodeindices;
        double const * node_voltages;
        double * node_diagonal;
        double * node_rhs;
        int nodecount;
    };


    struct top_local_ThreadVariables  {
        double * thread_data;

        double * gbl_ptr(size_t id) {
            return thread_data + 0 + (id % 1);
        }
        double & gbl(size_t id) {
            return thread_data[0 + (id % 1)];
        }

        top_local_ThreadVariables(double * const thread_data) {
            this->thread_data = thread_data;
        }
    };


    static top_local_Instance make_instance_top_local(_nrn_mechanism_cache_range* _lmc) {
        if(_lmc == nullptr) {
            return top_local_Instance();
        }

        return top_local_Instance {
            _lmc->template fpfield_ptr<0>(),
            _lmc->template fpfield_ptr<1>(),
            _lmc->template fpfield_ptr<2>(),
            _lmc->template fpfield_ptr<3>()
        };
    }


    static top_local_NodeData make_node_data_top_local(NrnThread& nt, Memb_list& _ml_arg) {
        return top_local_NodeData {
            _ml_arg.nodeindices,
            nt.node_voltage_storage(),
            nt.node_d_storage(),
            nt.node_rhs_storage(),
            _ml_arg.nodecount
        };
    }
    static top_local_NodeData make_node_data_top_local(Prop * _prop) {
        if(!_prop) {
            return top_local_NodeData();
        }

        static std::vector<int> node_index{0};
        Node* _node = _nrn_mechanism_access_node(_prop);
        return top_local_NodeData {
            node_index.data(),
            &_nrn_mechanism_access_voltage(_node),
            &_nrn_mechanism_access_d(_node),
            &_nrn_mechanism_access_rhs(_node),
            1
        };
    }

    static void nrn_destructor_top_local(Prop* prop);


    static void nrn_alloc_top_local(Prop* _prop) {
        Datum *_ppvar = nullptr;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const _iml = 0;
        assert(_nrn_mechanism_get_num_vars(_prop) == 4);
        /*initialize range parameters*/
    }


    /* Mechanism procedures and functions */
    inline static double get_gbl_top_local(_nrn_mechanism_cache_range& _lmc, top_local_Instance& inst, top_local_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, top_local_ThreadVariables& _thread_vars, NrnThread* nt);
    inline static int set_gbl_top_local(_nrn_mechanism_cache_range& _lmc, top_local_Instance& inst, top_local_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, top_local_ThreadVariables& _thread_vars, NrnThread* nt, double _lvalue);
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
    static void _hoc_get_gbl();
    static double _npy_get_gbl(Prop* _prop);
    static void _hoc_set_gbl();
    static double _npy_set_gbl(Prop* _prop);


    /* connect user functions to hoc names */
    static VoidFunc hoc_intfunc[] = {
        {"setdata_top_local", _hoc_setdata},
        {"set_gbl_top_local", _hoc_set_gbl},
        {"get_gbl_top_local", _hoc_get_gbl},
        {nullptr, nullptr}
    };
    static NPyDirectMechFunc npy_direct_func_proc[] = {
        {"set_gbl", _npy_set_gbl},
        {"get_gbl", _npy_get_gbl},
        {nullptr, nullptr}
    };
    static void thread_mem_init(Datum* _thread)  {
        if(top_local_global.thread_data_in_use) {
            _thread[0] = {neuron::container::do_not_search, new double[1]{}};
        }
        else {
            _thread[0] = {neuron::container::do_not_search, top_local_global.thread_data};
            top_local_global.thread_data_in_use = 1;
        }
    }
    static void thread_mem_cleanup(Datum* _thread)  {
        double * _thread_data_ptr = _thread[0].get<double*>();
        if(_thread_data_ptr == top_local_global.thread_data) {
            top_local_global.thread_data_in_use = 0;
        }
        else {
            delete[] _thread_data_ptr;
        }
    }
    static void _hoc_set_gbl() {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        Prop* _local_prop = _prop_id ? _extcall_prop : nullptr;
        _nrn_mechanism_cache_instance _lmc{_local_prop};
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_top_local(_local_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_top_local(_local_prop);
        auto _thread_vars = top_local_ThreadVariables(_thread[0].get<double*>());
        double _r = 0.0;
        _r = 1.;
        set_gbl_top_local(_lmc, inst, node_data, id, _ppvar, _thread, _thread_vars, nt, *getarg(1));
        hoc_retpushx(_r);
    }
    static double _npy_set_gbl(Prop* _prop) {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id = 0;
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_top_local(_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_top_local(_prop);
        auto _thread_vars = top_local_ThreadVariables(_thread[0].get<double*>());
        double _r = 0.0;
        _r = 1.;
        set_gbl_top_local(_lmc, inst, node_data, id, _ppvar, _thread, _thread_vars, nt, *getarg(1));
        return(_r);
    }
    static void _hoc_get_gbl() {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        Prop* _local_prop = _prop_id ? _extcall_prop : nullptr;
        _nrn_mechanism_cache_instance _lmc{_local_prop};
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_top_local(_local_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_top_local(_local_prop);
        auto _thread_vars = top_local_ThreadVariables(_thread[0].get<double*>());
        double _r = 0.0;
        _r = get_gbl_top_local(_lmc, inst, node_data, id, _ppvar, _thread, _thread_vars, nt);
        hoc_retpushx(_r);
    }
    static double _npy_get_gbl(Prop* _prop) {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id = 0;
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_top_local(_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_top_local(_prop);
        auto _thread_vars = top_local_ThreadVariables(_thread[0].get<double*>());
        double _r = 0.0;
        _r = get_gbl_top_local(_lmc, inst, node_data, id, _ppvar, _thread, _thread_vars, nt);
        return(_r);
    }


    inline int set_gbl_top_local(_nrn_mechanism_cache_range& _lmc, top_local_Instance& inst, top_local_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, top_local_ThreadVariables& _thread_vars, NrnThread* nt, double _lvalue) {
        int ret_set_gbl = 0;
        double v = node_data.node_voltages ? node_data.node_voltages[node_data.nodeindices[id]] : 0.0;
        _thread_vars.gbl(id) = _lvalue;
        return ret_set_gbl;
    }


    inline double get_gbl_top_local(_nrn_mechanism_cache_range& _lmc, top_local_Instance& inst, top_local_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, top_local_ThreadVariables& _thread_vars, NrnThread* nt) {
        double ret_get_gbl = 0.0;
        double v = node_data.node_voltages ? node_data.node_voltages[node_data.nodeindices[id]] : 0.0;
        ret_get_gbl = _thread_vars.gbl(id);
        return ret_get_gbl;
    }


    static void nrn_init_top_local(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_top_local(&_lmc);
        auto node_data = make_node_data_top_local(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto _thread_vars = top_local_ThreadVariables(_thread[0].get<double*>());
        auto nodecount = _ml_arg->nodecount;
        #pragma omp simd
        #pragma ivdep
        for (int id = 0; id < nodecount; id++) {
            auto* _ppvar = _ml_arg->pdata[id];
            int node_id = node_data.nodeindices[id];
            inst.v_unused[id] = node_data.node_voltages[node_id];
            _thread_vars.gbl(id) = 2.0;
        }
    }


    static inline double nrn_current_top_local(_nrn_mechanism_cache_range& _lmc, NrnThread* nt, Datum* _ppvar, Datum* _thread, top_local_ThreadVariables& _thread_vars, size_t id, top_local_Instance& inst, top_local_NodeData& node_data, double v) {
        inst.v_unused[id] = v;
        double current = 0.0;
        if (nt->_t > 0.33) {
            _thread_vars.gbl(id) = 3.0;
        }
        inst.y[id] = _thread_vars.gbl(id);
        inst.il[id] = 0.0000001 * (inst.v_unused[id] - 10.0);
        current += inst.il[id];
        return current;
    }


    /** update current */
    static void nrn_cur_top_local(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_top_local(&_lmc);
        auto node_data = make_node_data_top_local(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto _thread_vars = top_local_ThreadVariables(_thread[0].get<double*>());
        auto nodecount = _ml_arg->nodecount;
        #pragma omp simd
        #pragma ivdep
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_data.nodeindices[id];
            double v = node_data.node_voltages[node_id];
            auto* _ppvar = _ml_arg->pdata[id];
            double I1 = nrn_current_top_local(_lmc, nt, _ppvar, _thread, _thread_vars, id, inst, node_data, v+0.001);
            double I0 = nrn_current_top_local(_lmc, nt, _ppvar, _thread, _thread_vars, id, inst, node_data, v);
            double rhs = I0;
            double g = (I1-I0)/0.001;
            node_data.node_rhs[node_id] -= rhs;
            inst.g_unused[id] = g;
        }
    }


    static void nrn_state_top_local(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_top_local(&_lmc);
        auto node_data = make_node_data_top_local(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto _thread_vars = top_local_ThreadVariables(_thread[0].get<double*>());
        auto nodecount = _ml_arg->nodecount;
        #pragma omp simd
        #pragma ivdep
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_data.nodeindices[id];
            auto* _ppvar = _ml_arg->pdata[id];
            inst.v_unused[id] = node_data.node_voltages[node_id];
        }
    }


    static void nrn_jacob_top_local(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_top_local(&_lmc);
        auto node_data = make_node_data_top_local(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto _thread_vars = top_local_ThreadVariables(_thread[0].get<double*>());
        auto nodecount = _ml_arg->nodecount;
        #pragma omp simd
        #pragma ivdep
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_data.nodeindices[id];
            node_data.node_diagonal[node_id] += inst.g_unused[id];
        }
    }
    static void nrn_destructor_top_local(Prop* prop) {
        Datum* _ppvar = _nrn_mechanism_access_dparam(prop);
        _nrn_mechanism_cache_instance _lmc{prop};
        const size_t id = 0;
        auto inst = make_instance_top_local(prop ? &_lmc : nullptr);
        auto node_data = make_node_data_top_local(prop);
        auto _thread_vars = top_local_ThreadVariables(top_local_global.thread_data);

    }


    static void _initlists() {
    }


    extern "C" void _top_local_reg() {
        _initlists();

        register_mech(mechanism_info, nrn_alloc_top_local, nrn_cur_top_local, nrn_jacob_top_local, nrn_state_top_local, nrn_init_top_local, -1, 2);
        _extcall_thread.resize(2);
        thread_mem_init(_extcall_thread.data());
        top_local_global.thread_data_in_use = 0;

        mech_type = nrn_get_mechtype(mechanism_info[1]);
        hoc_register_parm_default(mech_type, &_parameter_defaults);
        _nrn_mechanism_register_data_fields(mech_type,
            _nrn_mechanism_field<double>{"y"} /* 0 */,
            _nrn_mechanism_field<double>{"il"} /* 1 */,
            _nrn_mechanism_field<double>{"v_unused"} /* 2 */,
            _nrn_mechanism_field<double>{"g_unused"} /* 3 */
        );

        hoc_register_prop_size(mech_type, 4, 0);
        hoc_register_var(hoc_scalar_double, hoc_vector_double, hoc_intfunc);
        hoc_register_npy_direct(mech_type, npy_direct_func_proc);
        _nrn_thread_reg(mech_type, 1, thread_mem_init);
        _nrn_thread_reg(mech_type, 0, thread_mem_cleanup);
    }
}
