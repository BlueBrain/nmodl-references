/*********************************************************
Model Name      : procedures
Filename        : procedures.mod
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
extern void _cvode_abstol(Symbol**, double*, int);
extern Node* nrn_alloc_node_;


namespace neuron {
    #ifndef NRN_PRCELLSTATE
    #define NRN_PRCELLSTATE 0
    #endif


    /** channel information */
    static const char *mechanism_info[] = {
        "7.7.0",
        "procedures",
        0,
        "x_procedures",
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
    struct procedures_Store {
    };
    static_assert(std::is_trivially_copy_constructible_v<procedures_Store>);
    static_assert(std::is_trivially_move_constructible_v<procedures_Store>);
    static_assert(std::is_trivially_copy_assignable_v<procedures_Store>);
    static_assert(std::is_trivially_move_assignable_v<procedures_Store>);
    static_assert(std::is_trivially_destructible_v<procedures_Store>);
    procedures_Store procedures_global;
    static std::vector<double> _parameter_defaults = {
    };


    /** all mechanism instance variables and global variables */
    struct procedures_Instance  {
        double* x{};
        double* v_unused{};
        procedures_Store* global{&procedures_global};
    };


    struct procedures_NodeData  {
        int const * nodeindices;
        double const * node_voltages;
        double * node_diagonal;
        double * node_rhs;
        int nodecount;
    };


    static procedures_Instance make_instance_procedures(_nrn_mechanism_cache_range& _lmc) {
        return procedures_Instance {
            _lmc.template fpfield_ptr<0>(),
            _lmc.template fpfield_ptr<1>()
        };
    }


    static procedures_NodeData make_node_data_procedures(NrnThread& nt, Memb_list& _ml_arg) {
        return procedures_NodeData {
            _ml_arg.nodeindices,
            nt.node_voltage_storage(),
            nt.node_d_storage(),
            nt.node_rhs_storage(),
            _ml_arg.nodecount
        };
    }
    static procedures_NodeData make_node_data_procedures(Prop * _prop) {
        static std::vector<int> node_index{0};
        Node* _node = _nrn_mechanism_access_node(_prop);
        return procedures_NodeData {
            node_index.data(),
            &_nrn_mechanism_access_voltage(_node),
            &_nrn_mechanism_access_d(_node),
            &_nrn_mechanism_access_rhs(_node),
            1
        };
    }

    void nrn_destructor_procedures(Prop* prop);


    static void nrn_alloc_procedures(Prop* _prop) {
        Datum *_ppvar = nullptr;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const _iml = 0;
        assert(_nrn_mechanism_get_num_vars(_prop) == 2);
        /*initialize range parameters*/
    }


    /* Mechanism procedures and functions */
    inline double identity_procedures(_nrn_mechanism_cache_range& _lmc, procedures_Instance& inst, procedures_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _lv);
    inline int set_x_42_procedures(_nrn_mechanism_cache_range& _lmc, procedures_Instance& inst, procedures_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt);
    inline int set_x_a_procedures(_nrn_mechanism_cache_range& _lmc, procedures_Instance& inst, procedures_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _la);
    inline int set_a_x_procedures(_nrn_mechanism_cache_range& _lmc, procedures_Instance& inst, procedures_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt);
    inline int set_x_v_procedures(_nrn_mechanism_cache_range& _lmc, procedures_Instance& inst, procedures_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt);
    inline int set_x_just_v_procedures(_nrn_mechanism_cache_range& _lmc, procedures_Instance& inst, procedures_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt);
    inline int set_x_just_vv_procedures(_nrn_mechanism_cache_range& _lmc, procedures_Instance& inst, procedures_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _lv);
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
    static void _hoc_identity(void);
    static double _npy_identity(Prop*);
    static void _hoc_set_x_42(void);
    static double _npy_set_x_42(Prop*);
    static void _hoc_set_x_a(void);
    static double _npy_set_x_a(Prop*);
    static void _hoc_set_a_x(void);
    static double _npy_set_a_x(Prop*);
    static void _hoc_set_x_v(void);
    static double _npy_set_x_v(Prop*);
    static void _hoc_set_x_just_v(void);
    static double _npy_set_x_just_v(Prop*);
    static void _hoc_set_x_just_vv(void);
    static double _npy_set_x_just_vv(Prop*);


    /* connect user functions to hoc names */
    static VoidFunc hoc_intfunc[] = {
        {"setdata_procedures", _hoc_setdata},
        {"set_x_42_procedures", _hoc_set_x_42},
        {"set_x_a_procedures", _hoc_set_x_a},
        {"set_a_x_procedures", _hoc_set_a_x},
        {"set_x_v_procedures", _hoc_set_x_v},
        {"set_x_just_v_procedures", _hoc_set_x_just_v},
        {"set_x_just_vv_procedures", _hoc_set_x_just_vv},
        {"identity_procedures", _hoc_identity},
        {nullptr, nullptr}
    };
    static NPyDirectMechFunc npy_direct_func_proc[] = {
        {"set_x_42", _npy_set_x_42},
        {"set_x_a", _npy_set_x_a},
        {"set_a_x", _npy_set_a_x},
        {"set_x_v", _npy_set_x_v},
        {"set_x_just_v", _npy_set_x_just_v},
        {"set_x_just_vv", _npy_set_x_just_vv},
        {"identity", _npy_identity},
        {nullptr, nullptr}
    };
    static void _hoc_set_x_42(void) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        if (!_prop_id) {
            hoc_execerror("No data for set_x_42_procedures. Requires prior call to setdata_procedures and that the specified mechanism instance still be in existence.", nullptr);
        }
        Prop* _local_prop = _extcall_prop;
        _nrn_mechanism_cache_instance _lmc{_local_prop};
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_procedures(_lmc);
        auto node_data = make_node_data_procedures(_local_prop);
        _r = 1.;
        set_x_42_procedures(_lmc, inst, node_data, id, _ppvar, _thread, nt);
        hoc_retpushx(_r);
    }
    static double _npy_set_x_42(Prop* _prop) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id = 0;
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_procedures(_lmc);
        auto node_data = make_node_data_procedures(_prop);
        _r = 1.;
        set_x_42_procedures(_lmc, inst, node_data, id, _ppvar, _thread, nt);
        return(_r);
    }
    static void _hoc_set_x_a(void) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        if (!_prop_id) {
            hoc_execerror("No data for set_x_a_procedures. Requires prior call to setdata_procedures and that the specified mechanism instance still be in existence.", nullptr);
        }
        Prop* _local_prop = _extcall_prop;
        _nrn_mechanism_cache_instance _lmc{_local_prop};
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_procedures(_lmc);
        auto node_data = make_node_data_procedures(_local_prop);
        _r = 1.;
        set_x_a_procedures(_lmc, inst, node_data, id, _ppvar, _thread, nt, *getarg(1));
        hoc_retpushx(_r);
    }
    static double _npy_set_x_a(Prop* _prop) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id = 0;
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_procedures(_lmc);
        auto node_data = make_node_data_procedures(_prop);
        _r = 1.;
        set_x_a_procedures(_lmc, inst, node_data, id, _ppvar, _thread, nt, *getarg(1));
        return(_r);
    }
    static void _hoc_set_a_x(void) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        if (!_prop_id) {
            hoc_execerror("No data for set_a_x_procedures. Requires prior call to setdata_procedures and that the specified mechanism instance still be in existence.", nullptr);
        }
        Prop* _local_prop = _extcall_prop;
        _nrn_mechanism_cache_instance _lmc{_local_prop};
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_procedures(_lmc);
        auto node_data = make_node_data_procedures(_local_prop);
        _r = 1.;
        set_a_x_procedures(_lmc, inst, node_data, id, _ppvar, _thread, nt);
        hoc_retpushx(_r);
    }
    static double _npy_set_a_x(Prop* _prop) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id = 0;
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_procedures(_lmc);
        auto node_data = make_node_data_procedures(_prop);
        _r = 1.;
        set_a_x_procedures(_lmc, inst, node_data, id, _ppvar, _thread, nt);
        return(_r);
    }
    static void _hoc_set_x_v(void) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        if (!_prop_id) {
            hoc_execerror("No data for set_x_v_procedures. Requires prior call to setdata_procedures and that the specified mechanism instance still be in existence.", nullptr);
        }
        Prop* _local_prop = _extcall_prop;
        _nrn_mechanism_cache_instance _lmc{_local_prop};
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_procedures(_lmc);
        auto node_data = make_node_data_procedures(_local_prop);
        _r = 1.;
        set_x_v_procedures(_lmc, inst, node_data, id, _ppvar, _thread, nt);
        hoc_retpushx(_r);
    }
    static double _npy_set_x_v(Prop* _prop) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id = 0;
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_procedures(_lmc);
        auto node_data = make_node_data_procedures(_prop);
        _r = 1.;
        set_x_v_procedures(_lmc, inst, node_data, id, _ppvar, _thread, nt);
        return(_r);
    }
    static void _hoc_set_x_just_v(void) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        if (!_prop_id) {
            hoc_execerror("No data for set_x_just_v_procedures. Requires prior call to setdata_procedures and that the specified mechanism instance still be in existence.", nullptr);
        }
        Prop* _local_prop = _extcall_prop;
        _nrn_mechanism_cache_instance _lmc{_local_prop};
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_procedures(_lmc);
        auto node_data = make_node_data_procedures(_local_prop);
        _r = 1.;
        set_x_just_v_procedures(_lmc, inst, node_data, id, _ppvar, _thread, nt);
        hoc_retpushx(_r);
    }
    static double _npy_set_x_just_v(Prop* _prop) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id = 0;
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_procedures(_lmc);
        auto node_data = make_node_data_procedures(_prop);
        _r = 1.;
        set_x_just_v_procedures(_lmc, inst, node_data, id, _ppvar, _thread, nt);
        return(_r);
    }
    static void _hoc_set_x_just_vv(void) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        if (!_prop_id) {
            hoc_execerror("No data for set_x_just_vv_procedures. Requires prior call to setdata_procedures and that the specified mechanism instance still be in existence.", nullptr);
        }
        Prop* _local_prop = _extcall_prop;
        _nrn_mechanism_cache_instance _lmc{_local_prop};
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_procedures(_lmc);
        auto node_data = make_node_data_procedures(_local_prop);
        _r = 1.;
        set_x_just_vv_procedures(_lmc, inst, node_data, id, _ppvar, _thread, nt, *getarg(1));
        hoc_retpushx(_r);
    }
    static double _npy_set_x_just_vv(Prop* _prop) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id = 0;
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_procedures(_lmc);
        auto node_data = make_node_data_procedures(_prop);
        _r = 1.;
        set_x_just_vv_procedures(_lmc, inst, node_data, id, _ppvar, _thread, nt, *getarg(1));
        return(_r);
    }
    static void _hoc_identity(void) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        Prop* _local_prop = _prop_id ? _extcall_prop : nullptr;
        _nrn_mechanism_cache_instance _lmc{_local_prop};
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_procedures(_lmc);
        auto node_data = make_node_data_procedures(_local_prop);
        _r = identity_procedures(_lmc, inst, node_data, id, _ppvar, _thread, nt, *getarg(1));
        hoc_retpushx(_r);
    }
    static double _npy_identity(Prop* _prop) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id = 0;
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_procedures(_lmc);
        auto node_data = make_node_data_procedures(_prop);
        _r = identity_procedures(_lmc, inst, node_data, id, _ppvar, _thread, nt, *getarg(1));
        return(_r);
    }


    inline int set_x_42_procedures(_nrn_mechanism_cache_range& _lmc, procedures_Instance& inst, procedures_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt) {
        int ret_set_x_42 = 0;
        auto v = node_data.node_voltages[node_data.nodeindices[id]];
        set_x_a_procedures(_lmc, inst, node_data, id, _ppvar, _thread, nt, 42.0);
        return ret_set_x_42;
    }


    inline int set_x_a_procedures(_nrn_mechanism_cache_range& _lmc, procedures_Instance& inst, procedures_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _la) {
        int ret_set_x_a = 0;
        auto v = node_data.node_voltages[node_data.nodeindices[id]];
        inst.x[id] = _la;
        return ret_set_x_a;
    }


    inline int set_a_x_procedures(_nrn_mechanism_cache_range& _lmc, procedures_Instance& inst, procedures_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt) {
        int ret_set_a_x = 0;
        auto v = node_data.node_voltages[node_data.nodeindices[id]];
        double a;
        a = inst.x[id];
        return ret_set_a_x;
    }


    inline int set_x_v_procedures(_nrn_mechanism_cache_range& _lmc, procedures_Instance& inst, procedures_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt) {
        int ret_set_x_v = 0;
        auto v = node_data.node_voltages[node_data.nodeindices[id]];
        inst.x[id] = v;
        return ret_set_x_v;
    }


    inline int set_x_just_v_procedures(_nrn_mechanism_cache_range& _lmc, procedures_Instance& inst, procedures_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt) {
        int ret_set_x_just_v = 0;
        auto v = node_data.node_voltages[node_data.nodeindices[id]];
        inst.x[id] = identity_procedures(_lmc, inst, node_data, id, _ppvar, _thread, nt, v);
        return ret_set_x_just_v;
    }


    inline int set_x_just_vv_procedures(_nrn_mechanism_cache_range& _lmc, procedures_Instance& inst, procedures_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _lv) {
        int ret_set_x_just_vv = 0;
        auto v = node_data.node_voltages[node_data.nodeindices[id]];
        inst.x[id] = identity_procedures(_lmc, inst, node_data, id, _ppvar, _thread, nt, _lv);
        return ret_set_x_just_vv;
    }


    inline double identity_procedures(_nrn_mechanism_cache_range& _lmc, procedures_Instance& inst, procedures_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _lv) {
        double ret_identity = 0.0;
        auto v = node_data.node_voltages[node_data.nodeindices[id]];
        ret_identity = _lv;
        return ret_identity;
    }


    void nrn_init_procedures(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_procedures(_lmc);
        auto node_data = make_node_data_procedures(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
            auto* _ppvar = _ml_arg->pdata[id];
            int node_id = node_data.nodeindices[id];
            auto v = node_data.node_voltages[node_id];
            set_a_x_procedures(_lmc, inst, node_data, id, _ppvar, _thread, nt);
        }
    }


    static void nrn_jacob_procedures(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_procedures(_lmc);
        auto node_data = make_node_data_procedures(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
        }
    }
    void nrn_destructor_procedures(Prop* prop) {
        Datum* _ppvar = _nrn_mechanism_access_dparam(prop);
        _nrn_mechanism_cache_instance _lmc{prop};
        const size_t id = 0;
        auto inst = make_instance_procedures(_lmc);
        auto node_data = make_node_data_procedures(prop);

    }


    static void _initlists() {
    }


    /** register channel with the simulator */
    extern "C" void _procedures_reg() {
        _initlists();

        register_mech(mechanism_info, nrn_alloc_procedures, nullptr, nullptr, nullptr, nrn_init_procedures, -1, 1);

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
