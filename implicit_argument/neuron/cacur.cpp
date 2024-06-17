/*********************************************************
Model Name      : cacur
Filename        : cacur.mod
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
#include <crout/crout.hpp>
#include <math.h>
#include <newton/newton.hpp>
#include <stdio.h>
#include <stdlib.h>

#include "mech_api.h"
#include "neuron/cache/mechanism_range.hpp"
#include "nrniv_mf.h"
#include "section_fwd.hpp"

/* NEURON global macro definitions */
/* VECTORIZED */
#define NRN_VECTORIZED 1

static constexpr auto number_of_datum_variables = 2;
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


namespace neuron {
    #ifndef NRN_PRCELLSTATE
    #define NRN_PRCELLSTATE 0
    #endif


    /** channel information */
    static const char *mechanism_info[] = {
        "7.7.0",
        "cacur",
        "del_cacur",
        "dur_cacur",
        "amp_cacur",
        0,
        0,
        0,
        0
    };


    /* NEURON global variables */
    static Symbol* _ca_sym;
    static int mech_type;
    static Prop* _extcall_prop;
    /* _prop_id kind of shadows _extcall_prop to allow validity checking. */
    static _nrn_non_owning_id_without_container _prop_id{};
    static int hoc_nrnpointerindex = -1;
    static _nrn_mechanism_std_vector<Datum> _extcall_thread;


    /** all global variables */
    struct cacur_Store {
    };
    static_assert(std::is_trivially_copy_constructible_v<cacur_Store>);
    static_assert(std::is_trivially_move_constructible_v<cacur_Store>);
    static_assert(std::is_trivially_copy_assignable_v<cacur_Store>);
    static_assert(std::is_trivially_move_assignable_v<cacur_Store>);
    static_assert(std::is_trivially_destructible_v<cacur_Store>);
    cacur_Store cacur_global;


    /** all mechanism instance variables and global variables */
    struct cacur_Instance  {
        double* del{};
        double* dur{};
        double* amp{};
        double* ica{};
        double* v_unused{};
        double* g_unused{};
        double* const* ion_ica{};
        double* const* ion_dicadv{};
        cacur_Store* global{&cacur_global};
    };


    struct cacur_NodeData  {
        int const * nodeindices;
        double const * node_voltages;
        double * node_diagonal;
        double * node_rhs;
        int nodecount;
    };


    static cacur_Instance make_instance_cacur(_nrn_mechanism_cache_range& _lmc) {
        return cacur_Instance {
            _lmc.template fpfield_ptr<0>(),
            _lmc.template fpfield_ptr<1>(),
            _lmc.template fpfield_ptr<2>(),
            _lmc.template fpfield_ptr<3>(),
            _lmc.template fpfield_ptr<4>(),
            _lmc.template fpfield_ptr<5>(),
            _lmc.template dptr_field_ptr<0>(),
            _lmc.template dptr_field_ptr<1>()
        };
    }


    static cacur_NodeData make_node_data_cacur(NrnThread& nt, Memb_list& _ml_arg) {
        return cacur_NodeData {
            _ml_arg.nodeindices,
            nt.node_voltage_storage(),
            nt.node_d_storage(),
            nt.node_rhs_storage(),
            _ml_arg.nodecount
        };
    }


    static void nrn_alloc_cacur(Prop* _prop) {
        Prop *prop_ion{};
        Datum *_ppvar{};
        _ppvar = nrn_prop_datum_alloc(mech_type, 2, _prop);
        _nrn_mechanism_access_dparam(_prop) = _ppvar;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const _iml{};
        assert(_nrn_mechanism_get_num_vars(_prop) == 6);
        /*initialize range parameters*/
        _lmc.template fpfield<0>(_iml) = 0; /* del */
        _lmc.template fpfield<1>(_iml) = 1; /* dur */
        _lmc.template fpfield<2>(_iml) = -1; /* amp */
        _nrn_mechanism_access_dparam(_prop) = _ppvar;
        Symbol * ca_sym = hoc_lookup("ca_ion");
        Prop * ca_prop = need_memb(ca_sym);
        _ppvar[0] = _nrn_mechanism_get_param_handle(ca_prop, 3);
        _ppvar[1] = _nrn_mechanism_get_param_handle(ca_prop, 4);
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
        {"setdata_cacur", _hoc_setdata},
        {nullptr, nullptr}
    };
    static NPyDirectMechFunc npy_direct_func_proc[] = {
        {nullptr, nullptr}
    };


    void nrn_init_cacur(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _type};
        auto inst = make_instance_cacur(_lmc);
        auto node_data = make_node_data_cacur(*nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* _thread = _ml_arg->_thread;
        for (int id = 0; id < nodecount; id++) {
            auto* _ppvar = _ml_arg->pdata[id];
            int node_id = node_data.nodeindices[id];
            auto v = node_data.node_voltages[node_id];
            inst.v_unused[id] = v;
        }
    }


    inline double nrn_current_cacur(_nrn_mechanism_cache_range& _lmc, NrnThread* nt, Datum* _ppvar, Datum* _thread, size_t id, cacur_Instance& inst, cacur_NodeData& node_data, double v) {
        double current = 0.0;
        if (inst.amp[id]) {
            at_time(nt, inst.del[id]);
            at_time(nt, inst.del[id] + inst.dur[id]);
        }
        if (nt->_t > inst.del[id] && nt->_t < inst.del[id] + inst.dur[id]) {
            inst.ica[id] = inst.amp[id];
        } else {
            inst.ica[id] = 0.0;
        }
        current += inst.ica[id];
        return current;
    }


    /** update current */
    void nrn_cur_cacur(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _type};
        auto inst = make_instance_cacur(_lmc);
        auto node_data = make_node_data_cacur(*nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* _thread = _ml_arg->_thread;
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_data.nodeindices[id];
            double v = node_data.node_voltages[node_id];
            auto* _ppvar = _ml_arg->pdata[id];
            double I1 = nrn_current_cacur(_lmc, nt, _ppvar, _thread, id, inst, node_data, v+0.001);
            double dica = inst.ica[id];
            double I0 = nrn_current_cacur(_lmc, nt, _ppvar, _thread, id, inst, node_data, v);
            double rhs = I0;
            double g = (I1-I0)/0.001;
            (*inst.ion_dicadv[id]) += (dica-inst.ica[id])/0.001;
            (*inst.ion_ica[id]) += inst.ica[id];
            node_data.node_rhs[node_id] -= rhs;
            inst.g_unused[id] = g;
        }
    }


    void nrn_state_cacur(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _type};
        auto inst = make_instance_cacur(_lmc);
        auto node_data = make_node_data_cacur(*nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* _thread = _ml_arg->_thread;
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_data.nodeindices[id];
            auto* _ppvar = _ml_arg->pdata[id];
            auto v = node_data.node_voltages[node_id];
        }
    }


    static void nrn_jacob_cacur(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _type};
        auto inst = make_instance_cacur(_lmc);
        auto node_data = make_node_data_cacur(*nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_data.nodeindices[id];
            node_data.node_diagonal[node_id] += inst.g_unused[id];
        }
    }


    static void _initlists() {
    }


    /** register channel with the simulator */
    extern "C" void _cacur_reg() {
        _initlists();

        ion_reg("ca", -10000.);

        _ca_sym = hoc_lookup("ca_ion");

        register_mech(mechanism_info, nrn_alloc_cacur, nrn_cur_cacur, nrn_jacob_cacur, nrn_state_cacur, nrn_init_cacur, hoc_nrnpointerindex, 1);

        mech_type = nrn_get_mechtype(mechanism_info[1]);
        _nrn_mechanism_register_data_fields(mech_type,
            _nrn_mechanism_field<double>{"del"} /* 0 */,
            _nrn_mechanism_field<double>{"dur"} /* 1 */,
            _nrn_mechanism_field<double>{"amp"} /* 2 */,
            _nrn_mechanism_field<double>{"ica"} /* 3 */,
            _nrn_mechanism_field<double>{"v_unused"} /* 4 */,
            _nrn_mechanism_field<double>{"g_unused"} /* 5 */,
            _nrn_mechanism_field<double*>{"ion_ica", "ca_ion"} /* 0 */,
            _nrn_mechanism_field<double*>{"ion_dicadv", "ca_ion"} /* 1 */
        );

        hoc_register_prop_size(mech_type, 6, 2);
        hoc_register_dparam_semantics(mech_type, 0, "ca_ion");
        hoc_register_dparam_semantics(mech_type, 1, "ca_ion");
        hoc_register_var(hoc_scalar_double, hoc_vector_double, hoc_intfunc);
        hoc_register_npy_direct(mech_type, npy_direct_func_proc);
    }
}
