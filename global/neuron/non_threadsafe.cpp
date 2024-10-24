/*********************************************************
Model Name      : non_threadsafe
Filename        : non_threadsafe.mod
NMODL Version   : 7.7.0
Vectorized      : false
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
/* NOT VECTORIZED */
#define NRN_VECTORIZED 0

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
        "non_threadsafe",
        0,
        0,
        "z_non_threadsafe",
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
    struct non_threadsafe_Store {
        double gbl{0};
        double parameter{41};
        double top_local{0};
        double z0{0};
    };
    static_assert(std::is_trivially_copy_constructible_v<non_threadsafe_Store>);
    static_assert(std::is_trivially_move_constructible_v<non_threadsafe_Store>);
    static_assert(std::is_trivially_copy_assignable_v<non_threadsafe_Store>);
    static_assert(std::is_trivially_move_assignable_v<non_threadsafe_Store>);
    static_assert(std::is_trivially_destructible_v<non_threadsafe_Store>);
    static non_threadsafe_Store non_threadsafe_global;
    auto gbl_non_threadsafe() -> std::decay<decltype(non_threadsafe_global.gbl)>::type  {
        return non_threadsafe_global.gbl;
    }
    auto parameter_non_threadsafe() -> std::decay<decltype(non_threadsafe_global.parameter)>::type  {
        return non_threadsafe_global.parameter;
    }
    auto top_local_non_threadsafe() -> std::decay<decltype(non_threadsafe_global.top_local)>::type  {
        return non_threadsafe_global.top_local;
    }
    auto z0_non_threadsafe() -> std::decay<decltype(non_threadsafe_global.z0)>::type  {
        return non_threadsafe_global.z0;
    }

    static std::vector<double> _parameter_defaults = {
    };


    /** all mechanism instance variables and global variables */
    struct non_threadsafe_Instance  {
        double* z{};
        double* Dz{};
        non_threadsafe_Store* global{&non_threadsafe_global};
    };


    struct non_threadsafe_NodeData  {
        int const * nodeindices;
        double const * node_voltages;
        double * node_diagonal;
        double * node_rhs;
        int nodecount;
    };


    static non_threadsafe_Instance make_instance_non_threadsafe(_nrn_mechanism_cache_range* _lmc) {
        if(_lmc == nullptr) {
            return non_threadsafe_Instance();
        }

        return non_threadsafe_Instance {
            _lmc->template fpfield_ptr<0>(),
            _lmc->template fpfield_ptr<1>()
        };
    }


    static non_threadsafe_NodeData make_node_data_non_threadsafe(NrnThread& nt, Memb_list& _ml_arg) {
        return non_threadsafe_NodeData {
            _ml_arg.nodeindices,
            nt.node_voltage_storage(),
            nt.node_d_storage(),
            nt.node_rhs_storage(),
            _ml_arg.nodecount
        };
    }
    static non_threadsafe_NodeData make_node_data_non_threadsafe(Prop * _prop) {
        if(!_prop) {
            return non_threadsafe_NodeData();
        }

        static std::vector<int> node_index{0};
        Node* _node = _nrn_mechanism_access_node(_prop);
        return non_threadsafe_NodeData {
            node_index.data(),
            &_nrn_mechanism_access_voltage(_node),
            &_nrn_mechanism_access_d(_node),
            &_nrn_mechanism_access_rhs(_node),
            1
        };
    }

    static void nrn_destructor_non_threadsafe(Prop* prop);


    static void nrn_alloc_non_threadsafe(Prop* _prop) {
        Datum *_ppvar = nullptr;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const _iml = 0;
        assert(_nrn_mechanism_get_num_vars(_prop) == 2);
        /*initialize range parameters*/
    }


    /* Mechanism procedures and functions */
    inline static double get_gbl_non_threadsafe(_nrn_mechanism_cache_range& _lmc, non_threadsafe_Instance& inst, non_threadsafe_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt);
    inline static double get_top_local_non_threadsafe(_nrn_mechanism_cache_range& _lmc, non_threadsafe_Instance& inst, non_threadsafe_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt);
    inline static double get_parameter_non_threadsafe(_nrn_mechanism_cache_range& _lmc, non_threadsafe_Instance& inst, non_threadsafe_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt);
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
        {"gbl_non_threadsafe", &non_threadsafe_global.gbl},
        {"parameter_non_threadsafe", &non_threadsafe_global.parameter},
        {nullptr, nullptr}
    };


    /** connect global (array) variables to hoc -- */
    static DoubVec hoc_vector_double[] = {
        {nullptr, nullptr, 0}
    };


    /* declaration of user functions */
    static void _hoc_get_gbl();
    static double _npy_get_gbl(Prop* _prop);
    static void _hoc_get_top_local();
    static double _npy_get_top_local(Prop* _prop);
    static void _hoc_get_parameter();
    static double _npy_get_parameter(Prop* _prop);


    /* connect user functions to hoc names */
    static VoidFunc hoc_intfunc[] = {
        {"setdata_non_threadsafe", _hoc_setdata},
        {"get_gbl_non_threadsafe", _hoc_get_gbl},
        {"get_top_local_non_threadsafe", _hoc_get_top_local},
        {"get_parameter_non_threadsafe", _hoc_get_parameter},
        {nullptr, nullptr}
    };
    static NPyDirectMechFunc npy_direct_func_proc[] = {
        {"get_gbl", _npy_get_gbl},
        {"get_top_local", _npy_get_top_local},
        {"get_parameter", _npy_get_parameter},
        {nullptr, nullptr}
    };
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
        auto inst = make_instance_non_threadsafe(_local_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_non_threadsafe(_local_prop);
        double _r = 0.0;
        _r = get_gbl_non_threadsafe(_lmc, inst, node_data, id, _ppvar, _thread, nt);
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
        auto inst = make_instance_non_threadsafe(_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_non_threadsafe(_prop);
        double _r = 0.0;
        _r = get_gbl_non_threadsafe(_lmc, inst, node_data, id, _ppvar, _thread, nt);
        return(_r);
    }
    static void _hoc_get_top_local() {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        Prop* _local_prop = _prop_id ? _extcall_prop : nullptr;
        _nrn_mechanism_cache_instance _lmc{_local_prop};
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_non_threadsafe(_local_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_non_threadsafe(_local_prop);
        double _r = 0.0;
        _r = get_top_local_non_threadsafe(_lmc, inst, node_data, id, _ppvar, _thread, nt);
        hoc_retpushx(_r);
    }
    static double _npy_get_top_local(Prop* _prop) {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id = 0;
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_non_threadsafe(_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_non_threadsafe(_prop);
        double _r = 0.0;
        _r = get_top_local_non_threadsafe(_lmc, inst, node_data, id, _ppvar, _thread, nt);
        return(_r);
    }
    static void _hoc_get_parameter() {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        Prop* _local_prop = _prop_id ? _extcall_prop : nullptr;
        _nrn_mechanism_cache_instance _lmc{_local_prop};
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_non_threadsafe(_local_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_non_threadsafe(_local_prop);
        double _r = 0.0;
        _r = get_parameter_non_threadsafe(_lmc, inst, node_data, id, _ppvar, _thread, nt);
        hoc_retpushx(_r);
    }
    static double _npy_get_parameter(Prop* _prop) {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id = 0;
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_non_threadsafe(_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_non_threadsafe(_prop);
        double _r = 0.0;
        _r = get_parameter_non_threadsafe(_lmc, inst, node_data, id, _ppvar, _thread, nt);
        return(_r);
    }


    inline double get_gbl_non_threadsafe(_nrn_mechanism_cache_range& _lmc, non_threadsafe_Instance& inst, non_threadsafe_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt) {
        double ret_get_gbl = 0.0;
        double v = node_data.node_voltages ? node_data.node_voltages[node_data.nodeindices[id]] : 0.0;
        ret_get_gbl = inst.global->gbl;
        return ret_get_gbl;
    }


    inline double get_top_local_non_threadsafe(_nrn_mechanism_cache_range& _lmc, non_threadsafe_Instance& inst, non_threadsafe_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt) {
        double ret_get_top_local = 0.0;
        double v = node_data.node_voltages ? node_data.node_voltages[node_data.nodeindices[id]] : 0.0;
        ret_get_top_local = inst.global->top_local;
        return ret_get_top_local;
    }


    inline double get_parameter_non_threadsafe(_nrn_mechanism_cache_range& _lmc, non_threadsafe_Instance& inst, non_threadsafe_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt) {
        double ret_get_parameter = 0.0;
        double v = node_data.node_voltages ? node_data.node_voltages[node_data.nodeindices[id]] : 0.0;
        ret_get_parameter = inst.global->parameter;
        return ret_get_parameter;
    }


    static void nrn_init_non_threadsafe(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_non_threadsafe(&_lmc);
        auto node_data = make_node_data_non_threadsafe(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
            auto* _ppvar = _ml_arg->pdata[id];
            int node_id = node_data.nodeindices[id];
            auto v = node_data.node_voltages[node_id];
            inst.z[id] = inst.global->z0;
            inst.global->gbl = 42.0;
            inst.global->top_local = 43.0;
        }
    }


    static void nrn_jacob_non_threadsafe(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_non_threadsafe(&_lmc);
        auto node_data = make_node_data_non_threadsafe(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
        }
    }
    static void nrn_destructor_non_threadsafe(Prop* prop) {
        Datum* _ppvar = _nrn_mechanism_access_dparam(prop);
        _nrn_mechanism_cache_instance _lmc{prop};
        const size_t id = 0;
        auto inst = make_instance_non_threadsafe(prop ? &_lmc : nullptr);
        auto node_data = make_node_data_non_threadsafe(prop);

    }


    static void _initlists() {
    }


    extern "C" void _non_threadsafe_reg() {
        _initlists();

        register_mech(mechanism_info, nrn_alloc_non_threadsafe, nullptr, nullptr, nullptr, nrn_init_non_threadsafe, -1, 1);

        mech_type = nrn_get_mechtype(mechanism_info[1]);
        hoc_register_parm_default(mech_type, &_parameter_defaults);
        _nrn_mechanism_register_data_fields(mech_type,
            _nrn_mechanism_field<double>{"z"} /* 0 */,
            _nrn_mechanism_field<double>{"Dz"} /* 1 */
        );

        hoc_register_prop_size(mech_type, 2, 0);
        hoc_register_var(hoc_scalar_double, hoc_vector_double, hoc_intfunc);
        hoc_register_npy_direct(mech_type, npy_direct_func_proc);
    }
}
