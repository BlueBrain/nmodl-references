/*********************************************************
Model Name      : basic_pointer
Filename        : basic_pointer.mod
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
extern void _cvode_abstol(Symbol**, double*, int);
extern Node* nrn_alloc_node_;


namespace neuron {
    #ifndef NRN_PRCELLSTATE
    #define NRN_PRCELLSTATE 0
    #endif


    /** channel information */
    static const char *mechanism_info[] = {
        "7.7.0",
        "basic_pointer",
        0,
        "x1_basic_pointer",
        "x2_basic_pointer",
        "ignore_basic_pointer",
        0,
        0,
        "p1_basic_pointer",
        "p2_basic_pointer",
        0
    };


    /* NEURON global variables */
    static Symbol* _ca_sym;
    static int mech_type;
    static Prop* _extcall_prop;
    /* _prop_id kind of shadows _extcall_prop to allow validity checking. */
    static _nrn_non_owning_id_without_container _prop_id{};
    static _nrn_mechanism_std_vector<Datum> _extcall_thread;


    /** all global variables */
    struct basic_pointer_Store {
    };
    static_assert(std::is_trivially_copy_constructible_v<basic_pointer_Store>);
    static_assert(std::is_trivially_move_constructible_v<basic_pointer_Store>);
    static_assert(std::is_trivially_copy_assignable_v<basic_pointer_Store>);
    static_assert(std::is_trivially_move_assignable_v<basic_pointer_Store>);
    static_assert(std::is_trivially_destructible_v<basic_pointer_Store>);
    static basic_pointer_Store basic_pointer_global;
    static std::vector<double> _parameter_defaults = {
    };


    /** all mechanism instance variables and global variables */
    struct basic_pointer_Instance  {
        double* x1{};
        double* x2{};
        double* ignore{};
        double* ica{};
        double* v_unused{};
        const double* const* ion_ica{};
        basic_pointer_Store* global{&basic_pointer_global};
    };


    struct basic_pointer_NodeData  {
        int const * nodeindices;
        double const * node_voltages;
        double * node_diagonal;
        double * node_rhs;
        int nodecount;
    };


    static basic_pointer_Instance make_instance_basic_pointer(_nrn_mechanism_cache_range* _lmc) {
        if(_lmc == nullptr) {
            return basic_pointer_Instance();
        }

        return basic_pointer_Instance {
            _lmc->template fpfield_ptr<0>(),
            _lmc->template fpfield_ptr<1>(),
            _lmc->template fpfield_ptr<2>(),
            _lmc->template fpfield_ptr<3>(),
            _lmc->template fpfield_ptr<4>(),
            _lmc->template dptr_field_ptr<0>()
        };
    }


    static basic_pointer_NodeData make_node_data_basic_pointer(NrnThread& nt, Memb_list& _ml_arg) {
        return basic_pointer_NodeData {
            _ml_arg.nodeindices,
            nt.node_voltage_storage(),
            nt.node_d_storage(),
            nt.node_rhs_storage(),
            _ml_arg.nodecount
        };
    }
    static basic_pointer_NodeData make_node_data_basic_pointer(Prop * _prop) {
        if(!_prop) {
            return basic_pointer_NodeData();
        }

        static std::vector<int> node_index{0};
        Node* _node = _nrn_mechanism_access_node(_prop);
        return basic_pointer_NodeData {
            node_index.data(),
            &_nrn_mechanism_access_voltage(_node),
            &_nrn_mechanism_access_d(_node),
            &_nrn_mechanism_access_rhs(_node),
            1
        };
    }

    static void nrn_destructor_basic_pointer(Prop* prop);


    static void nrn_alloc_basic_pointer(Prop* _prop) {
        Datum *_ppvar = nullptr;
        _ppvar = nrn_prop_datum_alloc(mech_type, 3, _prop);
        _nrn_mechanism_access_dparam(_prop) = _ppvar;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const _iml = 0;
        assert(_nrn_mechanism_get_num_vars(_prop) == 5);
        /*initialize range parameters*/
        _nrn_mechanism_access_dparam(_prop) = _ppvar;
        Symbol * ca_sym = hoc_lookup("ca_ion");
        Prop * ca_prop = need_memb(ca_sym);
        nrn_promote(ca_prop, 0, 0);
        _ppvar[0] = _nrn_mechanism_get_param_handle(ca_prop, 3);
    }


    /* Mechanism procedures and functions */
    inline static double read_p1_basic_pointer(_nrn_mechanism_cache_range& _lmc, basic_pointer_Instance& inst, basic_pointer_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt);
    inline static double read_p2_basic_pointer(_nrn_mechanism_cache_range& _lmc, basic_pointer_Instance& inst, basic_pointer_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt);
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
    static void _hoc_read_p1();
    static double _npy_read_p1(Prop* _prop);
    static void _hoc_read_p2();
    static double _npy_read_p2(Prop* _prop);


    /* connect user functions to hoc names */
    static VoidFunc hoc_intfunc[] = {
        {"setdata_basic_pointer", _hoc_setdata},
        {"read_p1_basic_pointer", _hoc_read_p1},
        {"read_p2_basic_pointer", _hoc_read_p2},
        {nullptr, nullptr}
    };
    static NPyDirectMechFunc npy_direct_func_proc[] = {
        {"read_p1", _npy_read_p1},
        {"read_p2", _npy_read_p2},
        {nullptr, nullptr}
    };
    static void _hoc_read_p1() {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        if (!_prop_id) {
            hoc_execerror("No data for read_p1_basic_pointer. Requires prior call to setdata_basic_pointer and that the specified mechanism instance still be in existence.", nullptr);
        }
        Prop* _local_prop = _extcall_prop;
        _nrn_mechanism_cache_instance _lmc{_local_prop};
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_basic_pointer(_local_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_basic_pointer(_local_prop);
        double _r = 0.0;
        _r = read_p1_basic_pointer(_lmc, inst, node_data, id, _ppvar, _thread, nt);
        hoc_retpushx(_r);
    }
    static double _npy_read_p1(Prop* _prop) {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id = 0;
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_basic_pointer(_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_basic_pointer(_prop);
        double _r = 0.0;
        _r = read_p1_basic_pointer(_lmc, inst, node_data, id, _ppvar, _thread, nt);
        return(_r);
    }
    static void _hoc_read_p2() {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        if (!_prop_id) {
            hoc_execerror("No data for read_p2_basic_pointer. Requires prior call to setdata_basic_pointer and that the specified mechanism instance still be in existence.", nullptr);
        }
        Prop* _local_prop = _extcall_prop;
        _nrn_mechanism_cache_instance _lmc{_local_prop};
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_basic_pointer(_local_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_basic_pointer(_local_prop);
        double _r = 0.0;
        _r = read_p2_basic_pointer(_lmc, inst, node_data, id, _ppvar, _thread, nt);
        hoc_retpushx(_r);
    }
    static double _npy_read_p2(Prop* _prop) {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id = 0;
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_basic_pointer(_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_basic_pointer(_prop);
        double _r = 0.0;
        _r = read_p2_basic_pointer(_lmc, inst, node_data, id, _ppvar, _thread, nt);
        return(_r);
    }


    inline double read_p1_basic_pointer(_nrn_mechanism_cache_range& _lmc, basic_pointer_Instance& inst, basic_pointer_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt) {
        double ret_read_p1 = 0.0;
        double v = node_data.node_voltages ? node_data.node_voltages[node_data.nodeindices[id]] : 0.0;
        ret_read_p1 = (*_ppvar[1].get<double*>());
        return ret_read_p1;
    }


    inline double read_p2_basic_pointer(_nrn_mechanism_cache_range& _lmc, basic_pointer_Instance& inst, basic_pointer_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt) {
        double ret_read_p2 = 0.0;
        double v = node_data.node_voltages ? node_data.node_voltages[node_data.nodeindices[id]] : 0.0;
        ret_read_p2 = (*_ppvar[2].get<double*>());
        return ret_read_p2;
    }


    static void nrn_init_basic_pointer(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_basic_pointer(&_lmc);
        auto node_data = make_node_data_basic_pointer(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
            auto* _ppvar = _ml_arg->pdata[id];
            int node_id = node_data.nodeindices[id];
            auto v = node_data.node_voltages[node_id];
            inst.ica[id] = (*inst.ion_ica[id]);
            inst.ignore[id] = inst.ica[id];
            inst.x1[id] = 0.0;
            inst.x2[id] = 0.0;
        }
    }


    static void nrn_jacob_basic_pointer(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_basic_pointer(&_lmc);
        auto node_data = make_node_data_basic_pointer(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
        }
    }
    static void nrn_destructor_basic_pointer(Prop* prop) {
        Datum* _ppvar = _nrn_mechanism_access_dparam(prop);
        _nrn_mechanism_cache_instance _lmc{prop};
        const size_t id = 0;
        auto inst = make_instance_basic_pointer(prop ? &_lmc : nullptr);
        auto node_data = make_node_data_basic_pointer(prop);

    }


    static void _initlists() {
    }


    extern "C" void _basic_pointer_reg() {
        _initlists();

        ion_reg("ca", -10000);

        _ca_sym = hoc_lookup("ca_ion");

        register_mech(mechanism_info, nrn_alloc_basic_pointer, nullptr, nullptr, nullptr, nrn_init_basic_pointer, 1, 1);

        mech_type = nrn_get_mechtype(mechanism_info[1]);
        hoc_register_parm_default(mech_type, &_parameter_defaults);
        _nrn_mechanism_register_data_fields(mech_type,
            _nrn_mechanism_field<double>{"x1"} /* 0 */,
            _nrn_mechanism_field<double>{"x2"} /* 1 */,
            _nrn_mechanism_field<double>{"ignore"} /* 2 */,
            _nrn_mechanism_field<double>{"ica"} /* 3 */,
            _nrn_mechanism_field<double>{"v_unused"} /* 4 */,
            _nrn_mechanism_field<double*>{"ion_ica", "ca_ion"} /* 0 */,
            _nrn_mechanism_field<double*>{"p1", "pointer"} /* 1 */,
            _nrn_mechanism_field<double*>{"p2", "pointer"} /* 2 */
        );

        hoc_register_prop_size(mech_type, 5, 3);
        hoc_register_dparam_semantics(mech_type, 0, "ca_ion");
        hoc_register_dparam_semantics(mech_type, 1, "pointer");
        hoc_register_dparam_semantics(mech_type, 2, "pointer");
        hoc_register_var(hoc_scalar_double, hoc_vector_double, hoc_intfunc);
        hoc_register_npy_direct(mech_type, npy_direct_func_proc);
    }
}
