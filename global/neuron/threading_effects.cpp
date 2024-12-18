/*********************************************************
Model Name      : threading_effects
Filename        : threading_effects.mod
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
extern void _nrn_thread_reg(int, int, void(*)(Datum*));

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
void _nrn_thread_table_reg(int, nrn_thread_table_check_t);
extern void _cvode_abstol(Symbol**, double*, int);
extern Node* nrn_alloc_node_;


namespace neuron {
    #ifndef NRN_PRCELLSTATE
    #define NRN_PRCELLSTATE 0
    #endif


    /** channel information */
    static const char *mechanism_info[] = {
        "7.7.0",
        "threading_effects",
        0,
        "y_threading_effects",
        "z_threading_effects",
        "il_threading_effects",
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
    struct threading_effects_Store {
        double thread_data_in_use{0};
        double thread_data[5] /* TODO init const-array */;
        double usetable{1};
        double tmin_compute_g_v1{};
        double mfac_compute_g_v1{};
        double t_g_v1[9]{};
    };
    static_assert(std::is_trivially_copy_constructible_v<threading_effects_Store>);
    static_assert(std::is_trivially_move_constructible_v<threading_effects_Store>);
    static_assert(std::is_trivially_copy_assignable_v<threading_effects_Store>);
    static_assert(std::is_trivially_move_assignable_v<threading_effects_Store>);
    static_assert(std::is_trivially_destructible_v<threading_effects_Store>);
    static threading_effects_Store threading_effects_global;
    auto thread_data_in_use_threading_effects() -> std::decay<decltype(threading_effects_global.thread_data_in_use)>::type  {
        return threading_effects_global.thread_data_in_use;
    }
    auto thread_data_threading_effects() -> std::decay<decltype(threading_effects_global.thread_data)>::type  {
        return threading_effects_global.thread_data;
    }
    auto usetable_threading_effects() -> std::decay<decltype(threading_effects_global.usetable)>::type  {
        return threading_effects_global.usetable;
    }
    auto tmin_compute_g_v1_threading_effects() -> std::decay<decltype(threading_effects_global.tmin_compute_g_v1)>::type  {
        return threading_effects_global.tmin_compute_g_v1;
    }
    auto mfac_compute_g_v1_threading_effects() -> std::decay<decltype(threading_effects_global.mfac_compute_g_v1)>::type  {
        return threading_effects_global.mfac_compute_g_v1;
    }
    auto t_g_v1_threading_effects() -> std::decay<decltype(threading_effects_global.t_g_v1)>::type  {
        return threading_effects_global.t_g_v1;
    }

    static std::vector<double> _parameter_defaults = {
    };


    /** all mechanism instance variables and global variables */
    struct threading_effects_Instance  {
        double* y{};
        double* z{};
        double* il{};
        double* v_unused{};
        double* g_unused{};
        threading_effects_Store* global{&threading_effects_global};
    };


    struct threading_effects_NodeData  {
        int const * nodeindices;
        double const * node_voltages;
        double * node_diagonal;
        double * node_rhs;
        int nodecount;
    };


    struct threading_effects_ThreadVariables  {
        double * thread_data;

        double * g_arr_ptr(size_t id) {
            return thread_data + 0 + (id % 1);
        }
        double & g_arr(size_t id) {
            return thread_data[0 + (id % 1)];
        }
        double * g_w_ptr(size_t id) {
            return thread_data + 3 + (id % 1);
        }
        double & g_w(size_t id) {
            return thread_data[3 + (id % 1)];
        }
        double * g_v1_ptr(size_t id) {
            return thread_data + 4 + (id % 1);
        }
        double & g_v1(size_t id) {
            return thread_data[4 + (id % 1)];
        }

        threading_effects_ThreadVariables(double * const thread_data) {
            this->thread_data = thread_data;
        }
    };


    static threading_effects_Instance make_instance_threading_effects(_nrn_mechanism_cache_range* _lmc) {
        if(_lmc == nullptr) {
            return threading_effects_Instance();
        }

        return threading_effects_Instance {
            _lmc->template fpfield_ptr<0>(),
            _lmc->template fpfield_ptr<1>(),
            _lmc->template fpfield_ptr<2>(),
            _lmc->template fpfield_ptr<3>(),
            _lmc->template fpfield_ptr<4>()
        };
    }


    static threading_effects_NodeData make_node_data_threading_effects(NrnThread& nt, Memb_list& _ml_arg) {
        return threading_effects_NodeData {
            _ml_arg.nodeindices,
            nt.node_voltage_storage(),
            nt.node_d_storage(),
            nt.node_rhs_storage(),
            _ml_arg.nodecount
        };
    }
    static threading_effects_NodeData make_node_data_threading_effects(Prop * _prop) {
        if(!_prop) {
            return threading_effects_NodeData();
        }

        static std::vector<int> node_index{0};
        Node* _node = _nrn_mechanism_access_node(_prop);
        return threading_effects_NodeData {
            node_index.data(),
            &_nrn_mechanism_access_voltage(_node),
            &_nrn_mechanism_access_d(_node),
            &_nrn_mechanism_access_rhs(_node),
            1
        };
    }

    static void nrn_destructor_threading_effects(Prop* prop);


    static void nrn_alloc_threading_effects(Prop* _prop) {
        Datum *_ppvar = nullptr;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const _iml = 0;
        assert(_nrn_mechanism_get_num_vars(_prop) == 5);
        /*initialize range parameters*/
    }


    /* Mechanism procedures and functions */
    inline static double sum_arr_threading_effects(_nrn_mechanism_cache_range& _lmc, threading_effects_Instance& inst, threading_effects_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, threading_effects_ThreadVariables& _thread_vars, NrnThread* nt);
    inline static int set_g_w_threading_effects(_nrn_mechanism_cache_range& _lmc, threading_effects_Instance& inst, threading_effects_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, threading_effects_ThreadVariables& _thread_vars, NrnThread* nt, double _lzz);
    inline static int compute_g_v1_threading_effects(_nrn_mechanism_cache_range& _lmc, threading_effects_Instance& inst, threading_effects_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, threading_effects_ThreadVariables& _thread_vars, NrnThread* nt, double _lzz);
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
    void update_table_compute_g_v1_threading_effects(_nrn_mechanism_cache_range& _lmc, threading_effects_Instance& inst, threading_effects_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, threading_effects_ThreadVariables& _thread_vars, NrnThread* nt);
    static void _check_table_thread(Memb_list* _ml, size_t id, Datum* _ppvar, Datum* _thread, double* _globals, NrnThread* nt, int _type, const _nrn_model_sorted_token& _sorted_token)
{
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml, _type};
        auto inst = make_instance_threading_effects(&_lmc);
        auto node_data = make_node_data_threading_effects(*nt, *_ml);
        auto _thread_vars = threading_effects_ThreadVariables(_thread[0].get<double*>());
        update_table_compute_g_v1_threading_effects(_lmc, inst, node_data, id, _ppvar, _thread, _thread_vars, nt);
    }


    /** connect global (scalar) variables to hoc -- */
    static DoubScal hoc_scalar_double[] = {
        {"usetable_threading_effects", &threading_effects_global.usetable},
        {"g_w_threading_effects", &threading_effects_global.thread_data[3]},
        {"g_v1_threading_effects", &threading_effects_global.thread_data[4]},
        {nullptr, nullptr}
    };


    /** connect global (array) variables to hoc -- */
    static DoubVec hoc_vector_double[] = {
        {"g_arr_threading_effects", (threading_effects_global.thread_data + 0), 3},
        {nullptr, nullptr, 0}
    };


    /* declaration of user functions */
    static void _hoc_sum_arr();
    static double _npy_sum_arr(Prop* _prop);
    static void _hoc_set_g_w();
    static double _npy_set_g_w(Prop* _prop);
    static void _hoc_compute_g_v1();
    static double _npy_compute_g_v1(Prop* _prop);


    /* connect user functions to hoc names */
    static VoidFunc hoc_intfunc[] = {
        {"setdata_threading_effects", _hoc_setdata},
        {"set_g_w_threading_effects", _hoc_set_g_w},
        {"compute_g_v1_threading_effects", _hoc_compute_g_v1},
        {"sum_arr_threading_effects", _hoc_sum_arr},
        {nullptr, nullptr}
    };
    static NPyDirectMechFunc npy_direct_func_proc[] = {
        {"set_g_w", _npy_set_g_w},
        {"compute_g_v1", _npy_compute_g_v1},
        {"sum_arr", _npy_sum_arr},
        {nullptr, nullptr}
    };
    static void thread_mem_init(Datum* _thread)  {
        if(threading_effects_global.thread_data_in_use) {
            _thread[0] = {neuron::container::do_not_search, new double[5]{}};
        }
        else {
            _thread[0] = {neuron::container::do_not_search, threading_effects_global.thread_data};
            threading_effects_global.thread_data_in_use = 1;
        }
    }
    static void thread_mem_cleanup(Datum* _thread)  {
        double * _thread_data_ptr = _thread[0].get<double*>();
        if(_thread_data_ptr == threading_effects_global.thread_data) {
            threading_effects_global.thread_data_in_use = 0;
        }
        else {
            delete[] _thread_data_ptr;
        }
    }
    static void _hoc_set_g_w() {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        Prop* _local_prop = _prop_id ? _extcall_prop : nullptr;
        _nrn_mechanism_cache_instance _lmc{_local_prop};
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_threading_effects(_local_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_threading_effects(_local_prop);
        auto _thread_vars = threading_effects_ThreadVariables(_thread[0].get<double*>());
        double _r = 0.0;
        _r = 1.;
        set_g_w_threading_effects(_lmc, inst, node_data, id, _ppvar, _thread, _thread_vars, nt, *getarg(1));
        hoc_retpushx(_r);
    }
    static double _npy_set_g_w(Prop* _prop) {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id = 0;
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_threading_effects(_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_threading_effects(_prop);
        auto _thread_vars = threading_effects_ThreadVariables(_thread[0].get<double*>());
        double _r = 0.0;
        _r = 1.;
        set_g_w_threading_effects(_lmc, inst, node_data, id, _ppvar, _thread, _thread_vars, nt, *getarg(1));
        return(_r);
    }
    static void _hoc_compute_g_v1() {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        Prop* _local_prop = _prop_id ? _extcall_prop : nullptr;
        _nrn_mechanism_cache_instance _lmc{_local_prop};
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_threading_effects(_local_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_threading_effects(_local_prop);
        auto _thread_vars = threading_effects_ThreadVariables(_thread[0].get<double*>());
        update_table_compute_g_v1_threading_effects(_lmc, inst, node_data, id, _ppvar, _thread, _thread_vars, nt);
        double _r = 0.0;
        _r = 1.;
        compute_g_v1_threading_effects(_lmc, inst, node_data, id, _ppvar, _thread, _thread_vars, nt, *getarg(1));
        hoc_retpushx(_r);
    }
    static double _npy_compute_g_v1(Prop* _prop) {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id = 0;
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_threading_effects(_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_threading_effects(_prop);
        auto _thread_vars = threading_effects_ThreadVariables(_thread[0].get<double*>());
        update_table_compute_g_v1_threading_effects(_lmc, inst, node_data, id, _ppvar, _thread, _thread_vars, nt);
        double _r = 0.0;
        _r = 1.;
        compute_g_v1_threading_effects(_lmc, inst, node_data, id, _ppvar, _thread, _thread_vars, nt, *getarg(1));
        return(_r);
    }
    static void _hoc_sum_arr() {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        Prop* _local_prop = _prop_id ? _extcall_prop : nullptr;
        _nrn_mechanism_cache_instance _lmc{_local_prop};
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_threading_effects(_local_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_threading_effects(_local_prop);
        auto _thread_vars = threading_effects_ThreadVariables(_thread[0].get<double*>());
        double _r = 0.0;
        _r = sum_arr_threading_effects(_lmc, inst, node_data, id, _ppvar, _thread, _thread_vars, nt);
        hoc_retpushx(_r);
    }
    static double _npy_sum_arr(Prop* _prop) {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id = 0;
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_threading_effects(_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_threading_effects(_prop);
        auto _thread_vars = threading_effects_ThreadVariables(_thread[0].get<double*>());
        double _r = 0.0;
        _r = sum_arr_threading_effects(_lmc, inst, node_data, id, _ppvar, _thread, _thread_vars, nt);
        return(_r);
    }


    inline int set_g_w_threading_effects(_nrn_mechanism_cache_range& _lmc, threading_effects_Instance& inst, threading_effects_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, threading_effects_ThreadVariables& _thread_vars, NrnThread* nt, double _lzz) {
        int ret_set_g_w = 0;
        double v = node_data.node_voltages ? node_data.node_voltages[node_data.nodeindices[id]] : 0.0;
        _thread_vars.g_w(id) = _lzz;
        return ret_set_g_w;
    }


    inline static int f_compute_g_v1_threading_effects(_nrn_mechanism_cache_range& _lmc, threading_effects_Instance& inst, threading_effects_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, threading_effects_ThreadVariables& _thread_vars, NrnThread* nt, double _lzz) {
        int ret_f_compute_g_v1 = 0;
        double v = node_data.node_voltages ? node_data.node_voltages[node_data.nodeindices[id]] : 0.0;
        _thread_vars.g_v1(id) = _lzz * _lzz;
        return ret_f_compute_g_v1;
    }


    void update_table_compute_g_v1_threading_effects(_nrn_mechanism_cache_range& _lmc, threading_effects_Instance& inst, threading_effects_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, threading_effects_ThreadVariables& _thread_vars, NrnThread* nt) {
        if (inst.global->usetable == 0) {
            return;
        }
        static bool make_table = true;
        if (make_table) {
            make_table = false;
            inst.global->tmin_compute_g_v1 = 3.0;
            double tmax = 4.0;
            double dx = (tmax-inst.global->tmin_compute_g_v1) / 8.;
            inst.global->mfac_compute_g_v1 = 1./dx;
            double x = inst.global->tmin_compute_g_v1;
            for (std::size_t i = 0; i < 9; x += dx, i++) {
                f_compute_g_v1_threading_effects(_lmc, inst, node_data, id, _ppvar, _thread, _thread_vars, nt, x);
                inst.global->t_g_v1[i] = _thread_vars.g_v1(id);
            }
        }
    }


    inline static int compute_g_v1_threading_effects(_nrn_mechanism_cache_range& _lmc, threading_effects_Instance& inst, threading_effects_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, threading_effects_ThreadVariables& _thread_vars, NrnThread* nt, double _lzz){
        if (inst.global->usetable == 0) {
            f_compute_g_v1_threading_effects(_lmc, inst, node_data, id, _ppvar, _thread, _thread_vars, nt, _lzz);
            return 0;
        }
        double xi = inst.global->mfac_compute_g_v1 * (_lzz - inst.global->tmin_compute_g_v1);
        if (isnan(xi)) {
            _thread_vars.g_v1(id) = xi;
            return 0;
        }
        if (xi <= 0. || xi >= 8.) {
            int index = (xi <= 0.) ? 0 : 8;
            _thread_vars.g_v1(id) = inst.global->t_g_v1[index];
            return 0;
        }
        int i = int(xi);
        double theta = xi - double(i);
        _thread_vars.g_v1(id) = inst.global->t_g_v1[i] + theta*(inst.global->t_g_v1[i+1]-inst.global->t_g_v1[i]);
        return 0;
    }


    inline double sum_arr_threading_effects(_nrn_mechanism_cache_range& _lmc, threading_effects_Instance& inst, threading_effects_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, threading_effects_ThreadVariables& _thread_vars, NrnThread* nt) {
        double ret_sum_arr = 0.0;
        double v = node_data.node_voltages ? node_data.node_voltages[node_data.nodeindices[id]] : 0.0;
        ret_sum_arr = (_thread_vars.g_arr_ptr(id))[static_cast<int>(0)] + (_thread_vars.g_arr_ptr(id))[static_cast<int>(1)] + (_thread_vars.g_arr_ptr(id))[static_cast<int>(2)];
        return ret_sum_arr;
    }


    static void nrn_init_threading_effects(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_threading_effects(&_lmc);
        auto node_data = make_node_data_threading_effects(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto _thread_vars = threading_effects_ThreadVariables(_thread[0].get<double*>());
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
            auto* _ppvar = _ml_arg->pdata[id];
            int node_id = node_data.nodeindices[id];
            inst.v_unused[id] = node_data.node_voltages[node_id];
            _thread_vars.g_w(id) = 48.0;
            _thread_vars.g_v1(id) = 0.0;
            (_thread_vars.g_arr_ptr(id))[static_cast<int>(0)] = 10.0 + inst.z[id];
            (_thread_vars.g_arr_ptr(id))[static_cast<int>(1)] = 10.1;
            (_thread_vars.g_arr_ptr(id))[static_cast<int>(2)] = 10.2;
            inst.y[id] = 10.0;
        }
    }


    static inline double nrn_current_threading_effects(_nrn_mechanism_cache_range& _lmc, NrnThread* nt, Datum* _ppvar, Datum* _thread, threading_effects_ThreadVariables& _thread_vars, size_t id, threading_effects_Instance& inst, threading_effects_NodeData& node_data, double v) {
        inst.v_unused[id] = v;
        double current = 0.0;
        if (nt->_t > 0.33) {
            _thread_vars.g_w(id) = sum_arr_threading_effects(_lmc, inst, node_data, id, _ppvar, _thread, _thread_vars, nt);
        }
        if (nt->_t > 0.66) {
            set_g_w_threading_effects(_lmc, inst, node_data, id, _ppvar, _thread, _thread_vars, nt, inst.z[id]);
            compute_g_v1_threading_effects(_lmc, inst, node_data, id, _ppvar, _thread, _thread_vars, nt, inst.z[id]);
        }
        inst.y[id] = _thread_vars.g_w(id) + _thread_vars.g_v1(id);
        inst.il[id] = 0.0000001 * (inst.v_unused[id] - 10.0);
        current += inst.il[id];
        return current;
    }


    /** update current */
    static void nrn_cur_threading_effects(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_threading_effects(&_lmc);
        auto node_data = make_node_data_threading_effects(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto _thread_vars = threading_effects_ThreadVariables(_thread[0].get<double*>());
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_data.nodeindices[id];
            double v = node_data.node_voltages[node_id];
            auto* _ppvar = _ml_arg->pdata[id];
            double I1 = nrn_current_threading_effects(_lmc, nt, _ppvar, _thread, _thread_vars, id, inst, node_data, v+0.001);
            double I0 = nrn_current_threading_effects(_lmc, nt, _ppvar, _thread, _thread_vars, id, inst, node_data, v);
            double rhs = I0;
            double g = (I1-I0)/0.001;
            node_data.node_rhs[node_id] -= rhs;
            inst.g_unused[id] = g;
        }
    }


    static void nrn_state_threading_effects(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_threading_effects(&_lmc);
        auto node_data = make_node_data_threading_effects(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto _thread_vars = threading_effects_ThreadVariables(_thread[0].get<double*>());
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_data.nodeindices[id];
            auto* _ppvar = _ml_arg->pdata[id];
            inst.v_unused[id] = node_data.node_voltages[node_id];
        }
    }


    static void nrn_jacob_threading_effects(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_threading_effects(&_lmc);
        auto node_data = make_node_data_threading_effects(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto _thread_vars = threading_effects_ThreadVariables(_thread[0].get<double*>());
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_data.nodeindices[id];
            node_data.node_diagonal[node_id] += inst.g_unused[id];
        }
    }
    static void nrn_destructor_threading_effects(Prop* prop) {
        Datum* _ppvar = _nrn_mechanism_access_dparam(prop);
        _nrn_mechanism_cache_instance _lmc{prop};
        const size_t id = 0;
        auto inst = make_instance_threading_effects(prop ? &_lmc : nullptr);
        auto node_data = make_node_data_threading_effects(prop);
        auto _thread_vars = threading_effects_ThreadVariables(threading_effects_global.thread_data);

    }


    static void _initlists() {
    }


    extern "C" void _threading_effects_reg() {
        _initlists();

        register_mech(mechanism_info, nrn_alloc_threading_effects, nrn_cur_threading_effects, nrn_jacob_threading_effects, nrn_state_threading_effects, nrn_init_threading_effects, -1, 2);
        _extcall_thread.resize(2);
        thread_mem_init(_extcall_thread.data());
        threading_effects_global.thread_data_in_use = 0;

        mech_type = nrn_get_mechtype(mechanism_info[1]);
        hoc_register_parm_default(mech_type, &_parameter_defaults);
        _nrn_thread_table_reg(mech_type, _check_table_thread);
        _nrn_mechanism_register_data_fields(mech_type,
            _nrn_mechanism_field<double>{"y"} /* 0 */,
            _nrn_mechanism_field<double>{"z"} /* 1 */,
            _nrn_mechanism_field<double>{"il"} /* 2 */,
            _nrn_mechanism_field<double>{"v_unused"} /* 3 */,
            _nrn_mechanism_field<double>{"g_unused"} /* 4 */
        );

        hoc_register_prop_size(mech_type, 5, 0);
        hoc_register_var(hoc_scalar_double, hoc_vector_double, hoc_intfunc);
        hoc_register_npy_direct(mech_type, npy_direct_func_proc);
        _nrn_thread_reg(mech_type, 1, thread_mem_init);
        _nrn_thread_reg(mech_type, 0, thread_mem_cleanup);
    }
}
