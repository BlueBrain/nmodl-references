/*********************************************************
Model Name      : localize_arguments
Filename        : localize_arguments.mod
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

static constexpr auto number_of_datum_variables = 3;
static constexpr auto number_of_floating_point_variables = 6;

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
        "localize_arguments",
        0,
        "x_localize_arguments",
        0,
        "s_localize_arguments",
        0,
        0
    };


    /* NEURON global variables */
    static Symbol* _na_sym;
    static int mech_type;
    static Prop* _extcall_prop;
    /* _prop_id kind of shadows _extcall_prop to allow validity checking. */
    static _nrn_non_owning_id_without_container _prop_id{};
    static _nrn_mechanism_std_vector<Datum> _extcall_thread;


    /** all global variables */
    struct localize_arguments_Store {
        double g{0};
        double p{42};
        double s0{0};
    };
    static_assert(std::is_trivially_copy_constructible_v<localize_arguments_Store>);
    static_assert(std::is_trivially_move_constructible_v<localize_arguments_Store>);
    static_assert(std::is_trivially_copy_assignable_v<localize_arguments_Store>);
    static_assert(std::is_trivially_move_assignable_v<localize_arguments_Store>);
    static_assert(std::is_trivially_destructible_v<localize_arguments_Store>);
    static localize_arguments_Store localize_arguments_global;
    auto g_localize_arguments() -> std::decay<decltype(localize_arguments_global.g)>::type  {
        return localize_arguments_global.g;
    }
    auto p_localize_arguments() -> std::decay<decltype(localize_arguments_global.p)>::type  {
        return localize_arguments_global.p;
    }
    auto s0_localize_arguments() -> std::decay<decltype(localize_arguments_global.s0)>::type  {
        return localize_arguments_global.s0;
    }

    static std::vector<double> _parameter_defaults = {
    };


    /** all mechanism instance variables and global variables */
    struct localize_arguments_Instance  {
        double* x{};
        double* s{};
        double* ina{};
        double* nai{};
        double* Ds{};
        double* v_unused{};
        const double* const* ion_ina{};
        const double* const* ion_nai{};
        const double* const* ion_nao{};
        localize_arguments_Store* global{&localize_arguments_global};
    };


    struct localize_arguments_NodeData  {
        int const * nodeindices;
        double const * node_voltages;
        double * node_diagonal;
        double * node_rhs;
        int nodecount;
    };


    static localize_arguments_Instance make_instance_localize_arguments(_nrn_mechanism_cache_range* _lmc) {
        if(_lmc == nullptr) {
            return localize_arguments_Instance();
        }

        return localize_arguments_Instance {
            _lmc->template fpfield_ptr<0>(),
            _lmc->template fpfield_ptr<1>(),
            _lmc->template fpfield_ptr<2>(),
            _lmc->template fpfield_ptr<3>(),
            _lmc->template fpfield_ptr<4>(),
            _lmc->template fpfield_ptr<5>(),
            _lmc->template dptr_field_ptr<0>(),
            _lmc->template dptr_field_ptr<1>(),
            _lmc->template dptr_field_ptr<2>()
        };
    }


    static localize_arguments_NodeData make_node_data_localize_arguments(NrnThread& nt, Memb_list& _ml_arg) {
        return localize_arguments_NodeData {
            _ml_arg.nodeindices,
            nt.node_voltage_storage(),
            nt.node_d_storage(),
            nt.node_rhs_storage(),
            _ml_arg.nodecount
        };
    }
    static localize_arguments_NodeData make_node_data_localize_arguments(Prop * _prop) {
        if(!_prop) {
            return localize_arguments_NodeData();
        }

        static std::vector<int> node_index{0};
        Node* _node = _nrn_mechanism_access_node(_prop);
        return localize_arguments_NodeData {
            node_index.data(),
            &_nrn_mechanism_access_voltage(_node),
            &_nrn_mechanism_access_d(_node),
            &_nrn_mechanism_access_rhs(_node),
            1
        };
    }

    static void nrn_destructor_localize_arguments(Prop* prop);


    static void nrn_alloc_localize_arguments(Prop* _prop) {
        Datum *_ppvar = nullptr;
        _ppvar = nrn_prop_datum_alloc(mech_type, 3, _prop);
        _nrn_mechanism_access_dparam(_prop) = _ppvar;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const _iml = 0;
        assert(_nrn_mechanism_get_num_vars(_prop) == 6);
        /*initialize range parameters*/
        _nrn_mechanism_access_dparam(_prop) = _ppvar;
        Symbol * na_sym = hoc_lookup("na_ion");
        Prop * na_prop = need_memb(na_sym);
        nrn_promote(na_prop, 1, 0);
        _ppvar[0] = _nrn_mechanism_get_param_handle(na_prop, 3);
        _ppvar[1] = _nrn_mechanism_get_param_handle(na_prop, 1);
        _ppvar[2] = _nrn_mechanism_get_param_handle(na_prop, 2);
    }


    /* Mechanism procedures and functions */
    inline static double id_v_localize_arguments(_nrn_mechanism_cache_range& _lmc, localize_arguments_Instance& inst, localize_arguments_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _lv);
    inline static double id_nai_localize_arguments(_nrn_mechanism_cache_range& _lmc, localize_arguments_Instance& inst, localize_arguments_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _lnai);
    inline static double id_ina_localize_arguments(_nrn_mechanism_cache_range& _lmc, localize_arguments_Instance& inst, localize_arguments_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _lina);
    inline static double id_x_localize_arguments(_nrn_mechanism_cache_range& _lmc, localize_arguments_Instance& inst, localize_arguments_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _lx);
    inline static double id_g_localize_arguments(_nrn_mechanism_cache_range& _lmc, localize_arguments_Instance& inst, localize_arguments_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _lg);
    inline static double id_s_localize_arguments(_nrn_mechanism_cache_range& _lmc, localize_arguments_Instance& inst, localize_arguments_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _ls);
    inline static double id_p_localize_arguments(_nrn_mechanism_cache_range& _lmc, localize_arguments_Instance& inst, localize_arguments_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _lp);
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
        {"g_localize_arguments", &localize_arguments_global.g},
        {"p_localize_arguments", &localize_arguments_global.p},
        {nullptr, nullptr}
    };


    /** connect global (array) variables to hoc -- */
    static DoubVec hoc_vector_double[] = {
        {nullptr, nullptr, 0}
    };


    /* declaration of user functions */
    static void _hoc_id_v();
    static double _npy_id_v(Prop* _prop);
    static void _hoc_id_nai();
    static double _npy_id_nai(Prop* _prop);
    static void _hoc_id_ina();
    static double _npy_id_ina(Prop* _prop);
    static void _hoc_id_x();
    static double _npy_id_x(Prop* _prop);
    static void _hoc_id_g();
    static double _npy_id_g(Prop* _prop);
    static void _hoc_id_s();
    static double _npy_id_s(Prop* _prop);
    static void _hoc_id_p();
    static double _npy_id_p(Prop* _prop);


    /* connect user functions to hoc names */
    static VoidFunc hoc_intfunc[] = {
        {"setdata_localize_arguments", _hoc_setdata},
        {"id_v_localize_arguments", _hoc_id_v},
        {"id_nai_localize_arguments", _hoc_id_nai},
        {"id_ina_localize_arguments", _hoc_id_ina},
        {"id_x_localize_arguments", _hoc_id_x},
        {"id_g_localize_arguments", _hoc_id_g},
        {"id_s_localize_arguments", _hoc_id_s},
        {"id_p_localize_arguments", _hoc_id_p},
        {nullptr, nullptr}
    };
    static NPyDirectMechFunc npy_direct_func_proc[] = {
        {"id_v", _npy_id_v},
        {"id_nai", _npy_id_nai},
        {"id_ina", _npy_id_ina},
        {"id_x", _npy_id_x},
        {"id_g", _npy_id_g},
        {"id_s", _npy_id_s},
        {"id_p", _npy_id_p},
        {nullptr, nullptr}
    };
    static void _hoc_id_v() {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        Prop* _local_prop = _prop_id ? _extcall_prop : nullptr;
        _nrn_mechanism_cache_instance _lmc{_local_prop};
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_localize_arguments(_local_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_localize_arguments(_local_prop);
        double _r = 0.0;
        _r = id_v_localize_arguments(_lmc, inst, node_data, id, _ppvar, _thread, nt, *getarg(1));
        hoc_retpushx(_r);
    }
    static double _npy_id_v(Prop* _prop) {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id = 0;
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_localize_arguments(_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_localize_arguments(_prop);
        double _r = 0.0;
        _r = id_v_localize_arguments(_lmc, inst, node_data, id, _ppvar, _thread, nt, *getarg(1));
        return(_r);
    }
    static void _hoc_id_nai() {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        Prop* _local_prop = _prop_id ? _extcall_prop : nullptr;
        _nrn_mechanism_cache_instance _lmc{_local_prop};
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_localize_arguments(_local_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_localize_arguments(_local_prop);
        double _r = 0.0;
        _r = id_nai_localize_arguments(_lmc, inst, node_data, id, _ppvar, _thread, nt, *getarg(1));
        hoc_retpushx(_r);
    }
    static double _npy_id_nai(Prop* _prop) {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id = 0;
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_localize_arguments(_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_localize_arguments(_prop);
        double _r = 0.0;
        _r = id_nai_localize_arguments(_lmc, inst, node_data, id, _ppvar, _thread, nt, *getarg(1));
        return(_r);
    }
    static void _hoc_id_ina() {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        Prop* _local_prop = _prop_id ? _extcall_prop : nullptr;
        _nrn_mechanism_cache_instance _lmc{_local_prop};
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_localize_arguments(_local_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_localize_arguments(_local_prop);
        double _r = 0.0;
        _r = id_ina_localize_arguments(_lmc, inst, node_data, id, _ppvar, _thread, nt, *getarg(1));
        hoc_retpushx(_r);
    }
    static double _npy_id_ina(Prop* _prop) {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id = 0;
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_localize_arguments(_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_localize_arguments(_prop);
        double _r = 0.0;
        _r = id_ina_localize_arguments(_lmc, inst, node_data, id, _ppvar, _thread, nt, *getarg(1));
        return(_r);
    }
    static void _hoc_id_x() {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        Prop* _local_prop = _prop_id ? _extcall_prop : nullptr;
        _nrn_mechanism_cache_instance _lmc{_local_prop};
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_localize_arguments(_local_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_localize_arguments(_local_prop);
        double _r = 0.0;
        _r = id_x_localize_arguments(_lmc, inst, node_data, id, _ppvar, _thread, nt, *getarg(1));
        hoc_retpushx(_r);
    }
    static double _npy_id_x(Prop* _prop) {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id = 0;
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_localize_arguments(_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_localize_arguments(_prop);
        double _r = 0.0;
        _r = id_x_localize_arguments(_lmc, inst, node_data, id, _ppvar, _thread, nt, *getarg(1));
        return(_r);
    }
    static void _hoc_id_g() {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        Prop* _local_prop = _prop_id ? _extcall_prop : nullptr;
        _nrn_mechanism_cache_instance _lmc{_local_prop};
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_localize_arguments(_local_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_localize_arguments(_local_prop);
        double _r = 0.0;
        _r = id_g_localize_arguments(_lmc, inst, node_data, id, _ppvar, _thread, nt, *getarg(1));
        hoc_retpushx(_r);
    }
    static double _npy_id_g(Prop* _prop) {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id = 0;
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_localize_arguments(_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_localize_arguments(_prop);
        double _r = 0.0;
        _r = id_g_localize_arguments(_lmc, inst, node_data, id, _ppvar, _thread, nt, *getarg(1));
        return(_r);
    }
    static void _hoc_id_s() {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        Prop* _local_prop = _prop_id ? _extcall_prop : nullptr;
        _nrn_mechanism_cache_instance _lmc{_local_prop};
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_localize_arguments(_local_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_localize_arguments(_local_prop);
        double _r = 0.0;
        _r = id_s_localize_arguments(_lmc, inst, node_data, id, _ppvar, _thread, nt, *getarg(1));
        hoc_retpushx(_r);
    }
    static double _npy_id_s(Prop* _prop) {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id = 0;
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_localize_arguments(_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_localize_arguments(_prop);
        double _r = 0.0;
        _r = id_s_localize_arguments(_lmc, inst, node_data, id, _ppvar, _thread, nt, *getarg(1));
        return(_r);
    }
    static void _hoc_id_p() {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        Prop* _local_prop = _prop_id ? _extcall_prop : nullptr;
        _nrn_mechanism_cache_instance _lmc{_local_prop};
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_localize_arguments(_local_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_localize_arguments(_local_prop);
        double _r = 0.0;
        _r = id_p_localize_arguments(_lmc, inst, node_data, id, _ppvar, _thread, nt, *getarg(1));
        hoc_retpushx(_r);
    }
    static double _npy_id_p(Prop* _prop) {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id = 0;
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_localize_arguments(_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_localize_arguments(_prop);
        double _r = 0.0;
        _r = id_p_localize_arguments(_lmc, inst, node_data, id, _ppvar, _thread, nt, *getarg(1));
        return(_r);
    }


    inline double id_v_localize_arguments(_nrn_mechanism_cache_range& _lmc, localize_arguments_Instance& inst, localize_arguments_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _lv) {
        double ret_id_v = 0.0;
        double v = node_data.node_voltages ? node_data.node_voltages[node_data.nodeindices[id]] : 0.0;
        ret_id_v = _lv;
        return ret_id_v;
    }


    inline double id_nai_localize_arguments(_nrn_mechanism_cache_range& _lmc, localize_arguments_Instance& inst, localize_arguments_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _lnai) {
        double ret_id_nai = 0.0;
        double v = node_data.node_voltages ? node_data.node_voltages[node_data.nodeindices[id]] : 0.0;
        ret_id_nai = _lnai;
        return ret_id_nai;
    }


    inline double id_ina_localize_arguments(_nrn_mechanism_cache_range& _lmc, localize_arguments_Instance& inst, localize_arguments_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _lina) {
        double ret_id_ina = 0.0;
        double v = node_data.node_voltages ? node_data.node_voltages[node_data.nodeindices[id]] : 0.0;
        ret_id_ina = _lina;
        return ret_id_ina;
    }


    inline double id_x_localize_arguments(_nrn_mechanism_cache_range& _lmc, localize_arguments_Instance& inst, localize_arguments_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _lx) {
        double ret_id_x = 0.0;
        double v = node_data.node_voltages ? node_data.node_voltages[node_data.nodeindices[id]] : 0.0;
        ret_id_x = _lx;
        return ret_id_x;
    }


    inline double id_g_localize_arguments(_nrn_mechanism_cache_range& _lmc, localize_arguments_Instance& inst, localize_arguments_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _lg) {
        double ret_id_g = 0.0;
        double v = node_data.node_voltages ? node_data.node_voltages[node_data.nodeindices[id]] : 0.0;
        ret_id_g = _lg;
        return ret_id_g;
    }


    inline double id_s_localize_arguments(_nrn_mechanism_cache_range& _lmc, localize_arguments_Instance& inst, localize_arguments_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _ls) {
        double ret_id_s = 0.0;
        double v = node_data.node_voltages ? node_data.node_voltages[node_data.nodeindices[id]] : 0.0;
        ret_id_s = _ls;
        return ret_id_s;
    }


    inline double id_p_localize_arguments(_nrn_mechanism_cache_range& _lmc, localize_arguments_Instance& inst, localize_arguments_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _lp) {
        double ret_id_p = 0.0;
        double v = node_data.node_voltages ? node_data.node_voltages[node_data.nodeindices[id]] : 0.0;
        ret_id_p = _lp;
        return ret_id_p;
    }


    static void nrn_init_localize_arguments(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_localize_arguments(&_lmc);
        auto node_data = make_node_data_localize_arguments(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
            auto* _ppvar = _ml_arg->pdata[id];
            int node_id = node_data.nodeindices[id];
            inst.v_unused[id] = node_data.node_voltages[node_id];
            inst.s[id] = inst.global->s0;
            inst.ina[id] = (*inst.ion_ina[id]);
            inst.nai[id] = (*inst.ion_nai[id]);
            inst.x[id] = 42.0;
            inst.s[id] = 42.0;
        }
    }


    static void nrn_jacob_localize_arguments(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_localize_arguments(&_lmc);
        auto node_data = make_node_data_localize_arguments(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
        }
    }
    static void nrn_destructor_localize_arguments(Prop* prop) {
        Datum* _ppvar = _nrn_mechanism_access_dparam(prop);
        _nrn_mechanism_cache_instance _lmc{prop};
        const size_t id = 0;
        auto inst = make_instance_localize_arguments(prop ? &_lmc : nullptr);
        auto node_data = make_node_data_localize_arguments(prop);

    }


    static void _initlists() {
    }


    extern "C" void _localize_arguments_reg() {
        _initlists();

        ion_reg("na", -10000);

        _na_sym = hoc_lookup("na_ion");

        register_mech(mechanism_info, nrn_alloc_localize_arguments, nullptr, nullptr, nullptr, nrn_init_localize_arguments, -1, 1);

        mech_type = nrn_get_mechtype(mechanism_info[1]);
        hoc_register_parm_default(mech_type, &_parameter_defaults);
        _nrn_mechanism_register_data_fields(mech_type,
            _nrn_mechanism_field<double>{"x"} /* 0 */,
            _nrn_mechanism_field<double>{"s"} /* 1 */,
            _nrn_mechanism_field<double>{"ina"} /* 2 */,
            _nrn_mechanism_field<double>{"nai"} /* 3 */,
            _nrn_mechanism_field<double>{"Ds"} /* 4 */,
            _nrn_mechanism_field<double>{"v_unused"} /* 5 */,
            _nrn_mechanism_field<double*>{"ion_ina", "na_ion"} /* 0 */,
            _nrn_mechanism_field<double*>{"ion_nai", "na_ion"} /* 1 */,
            _nrn_mechanism_field<double*>{"ion_nao", "na_ion"} /* 2 */
        );

        hoc_register_prop_size(mech_type, 6, 3);
        hoc_register_dparam_semantics(mech_type, 0, "na_ion");
        hoc_register_dparam_semantics(mech_type, 1, "na_ion");
        hoc_register_dparam_semantics(mech_type, 2, "na_ion");
        hoc_register_var(hoc_scalar_double, hoc_vector_double, hoc_intfunc);
        hoc_register_npy_direct(mech_type, npy_direct_func_proc);
    }
}
