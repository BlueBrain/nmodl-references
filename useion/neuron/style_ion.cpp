/*********************************************************
Model Name      : style_ion
Filename        : style_ion.mod
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

/* NEURON global macro definitions */
/* VECTORIZED */
#define NRN_VECTORIZED 1

static constexpr auto number_of_datum_variables = 7;
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
        "style_ion",
        0,
        0,
        0,
        0
    };


    /* NEURON global variables */
    static Symbol* _ca_sym;
    static Symbol* _na_sym;
    static int mech_type;
    static Prop* _extcall_prop;
    /* _prop_id kind of shadows _extcall_prop to allow validity checking. */
    static _nrn_non_owning_id_without_container _prop_id{};
    static _nrn_mechanism_std_vector<Datum> _extcall_thread;


    /** all global variables */
    struct style_ion_Store {
    };
    static_assert(std::is_trivially_copy_constructible_v<style_ion_Store>);
    static_assert(std::is_trivially_move_constructible_v<style_ion_Store>);
    static_assert(std::is_trivially_copy_assignable_v<style_ion_Store>);
    static_assert(std::is_trivially_move_assignable_v<style_ion_Store>);
    static_assert(std::is_trivially_destructible_v<style_ion_Store>);
    static style_ion_Store style_ion_global;
    static std::vector<double> _parameter_defaults = {
    };


    /** all mechanism instance variables and global variables */
    struct style_ion_Instance  {
        double* cai{};
        double* eca{};
        double* nai{};
        double* v_unused{};
        const double* const* ion_cao{};
        double* const* ion_cai{};
        double* const* ion_eca{};
        double* const* ion_ca_erev{};
        const double* const* ion_nai{};
        const double* const* ion_nao{};
        style_ion_Store* global{&style_ion_global};
    };


    struct style_ion_NodeData  {
        int const * nodeindices;
        double const * node_voltages;
        double * node_diagonal;
        double * node_rhs;
        int nodecount;
    };


    static style_ion_Instance make_instance_style_ion(_nrn_mechanism_cache_range* _lmc) {
        if(_lmc == nullptr) {
            return style_ion_Instance();
        }

        return style_ion_Instance {
            _lmc->template fpfield_ptr<0>(),
            _lmc->template fpfield_ptr<1>(),
            _lmc->template fpfield_ptr<2>(),
            _lmc->template fpfield_ptr<3>(),
            _lmc->template dptr_field_ptr<0>(),
            _lmc->template dptr_field_ptr<1>(),
            _lmc->template dptr_field_ptr<2>(),
            _lmc->template dptr_field_ptr<3>(),
            _lmc->template dptr_field_ptr<5>(),
            _lmc->template dptr_field_ptr<6>()
        };
    }


    static style_ion_NodeData make_node_data_style_ion(NrnThread& nt, Memb_list& _ml_arg) {
        return style_ion_NodeData {
            _ml_arg.nodeindices,
            nt.node_voltage_storage(),
            nt.node_d_storage(),
            nt.node_rhs_storage(),
            _ml_arg.nodecount
        };
    }
    static style_ion_NodeData make_node_data_style_ion(Prop * _prop) {
        if(!_prop) {
            return style_ion_NodeData();
        }

        static std::vector<int> node_index{0};
        Node* _node = _nrn_mechanism_access_node(_prop);
        return style_ion_NodeData {
            node_index.data(),
            &_nrn_mechanism_access_voltage(_node),
            &_nrn_mechanism_access_d(_node),
            &_nrn_mechanism_access_rhs(_node),
            1
        };
    }

    static void nrn_destructor_style_ion(Prop* prop);


    static void nrn_alloc_style_ion(Prop* _prop) {
        Datum *_ppvar = nullptr;
        _ppvar = nrn_prop_datum_alloc(mech_type, 7, _prop);
        _nrn_mechanism_access_dparam(_prop) = _ppvar;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const _iml = 0;
        assert(_nrn_mechanism_get_num_vars(_prop) == 4);
        /*initialize range parameters*/
        _nrn_mechanism_access_dparam(_prop) = _ppvar;
        Symbol * ca_sym = hoc_lookup("ca_ion");
        Prop * ca_prop = need_memb(ca_sym);
        nrn_check_conc_write(_prop, ca_prop, 1);
        nrn_promote(ca_prop, 3, 3);
        _ppvar[0] = _nrn_mechanism_get_param_handle(ca_prop, 2);
        _ppvar[1] = _nrn_mechanism_get_param_handle(ca_prop, 1);
        _ppvar[2] = _nrn_mechanism_get_param_handle(ca_prop, 0);
        _ppvar[3] = _nrn_mechanism_get_param_handle(ca_prop, 0);
        _ppvar[4] = {neuron::container::do_not_search, &(_nrn_mechanism_access_dparam(ca_prop)[0].literal_value<int>())};
        Symbol * na_sym = hoc_lookup("na_ion");
        Prop * na_prop = need_memb(na_sym);
        nrn_promote(na_prop, 1, 0);
        _ppvar[5] = _nrn_mechanism_get_param_handle(na_prop, 1);
        _ppvar[6] = _nrn_mechanism_get_param_handle(na_prop, 2);
    }


    /* Mechanism procedures and functions */
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


    /* connect user functions to hoc names */
    static VoidFunc hoc_intfunc[] = {
        {"setdata_style_ion", _hoc_setdata},
        {nullptr, nullptr}
    };
    static NPyDirectMechFunc npy_direct_func_proc[] = {
        {nullptr, nullptr}
    };


    static void nrn_init_style_ion(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_style_ion(&_lmc);
        auto node_data = make_node_data_style_ion(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
            auto* _ppvar = _ml_arg->pdata[id];
            int node_id = node_data.nodeindices[id];
            inst.v_unused[id] = node_data.node_voltages[node_id];
            inst.cai[id] = (*inst.ion_cai[id]);
            inst.nai[id] = (*inst.ion_nai[id]);
            inst.cai[id] = 42.0;
            (*inst.ion_cai[id]) = inst.cai[id];
            (*inst.ion_eca[id]) = inst.eca[id];
            int _style_ca = *(_ppvar[4].get<int*>());
            nrn_wrote_conc(_ca_sym, (*inst.ion_ca_erev[id]), (*inst.ion_cai[id]), (*inst.ion_cao[id]), _style_ca);
        }
    }


    static void nrn_jacob_style_ion(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_style_ion(&_lmc);
        auto node_data = make_node_data_style_ion(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
        }
    }
    static void nrn_destructor_style_ion(Prop* prop) {
        Datum* _ppvar = _nrn_mechanism_access_dparam(prop);
        _nrn_mechanism_cache_instance _lmc{prop};
        const size_t id = 0;
        auto inst = make_instance_style_ion(prop ? &_lmc : nullptr);
        auto node_data = make_node_data_style_ion(prop);

    }


    static void _initlists() {
    }


    extern "C" void _style_ion_reg() {
        _initlists();

        ion_reg("ca", -10000);
        ion_reg("na", -10000);

        _ca_sym = hoc_lookup("ca_ion");
        _na_sym = hoc_lookup("na_ion");

        register_mech(mechanism_info, nrn_alloc_style_ion, nullptr, nullptr, nullptr, nrn_init_style_ion, -1, 1);

        mech_type = nrn_get_mechtype(mechanism_info[1]);
        hoc_register_parm_default(mech_type, &_parameter_defaults);
        _nrn_mechanism_register_data_fields(mech_type,
            _nrn_mechanism_field<double>{"cai"} /* 0 */,
            _nrn_mechanism_field<double>{"eca"} /* 1 */,
            _nrn_mechanism_field<double>{"nai"} /* 2 */,
            _nrn_mechanism_field<double>{"v_unused"} /* 3 */,
            _nrn_mechanism_field<double*>{"ion_cao", "ca_ion"} /* 0 */,
            _nrn_mechanism_field<double*>{"ion_cai", "ca_ion"} /* 1 */,
            _nrn_mechanism_field<double*>{"ion_eca", "ca_ion"} /* 2 */,
            _nrn_mechanism_field<double*>{"ion_ca_erev", "ca_ion"} /* 3 */,
            _nrn_mechanism_field<int*>{"style_ca", "#ca_ion"} /* 4 */,
            _nrn_mechanism_field<double*>{"ion_nai", "na_ion"} /* 5 */,
            _nrn_mechanism_field<double*>{"ion_nao", "na_ion"} /* 6 */
        );

        hoc_register_prop_size(mech_type, 4, 7);
        hoc_register_dparam_semantics(mech_type, 0, "ca_ion");
        hoc_register_dparam_semantics(mech_type, 1, "ca_ion");
        hoc_register_dparam_semantics(mech_type, 2, "ca_ion");
        hoc_register_dparam_semantics(mech_type, 3, "ca_ion");
        hoc_register_dparam_semantics(mech_type, 4, "#ca_ion");
        hoc_register_dparam_semantics(mech_type, 5, "na_ion");
        hoc_register_dparam_semantics(mech_type, 6, "na_ion");
        nrn_writes_conc(mech_type, 0);
        hoc_register_var(hoc_scalar_double, hoc_vector_double, hoc_intfunc);
        hoc_register_npy_direct(mech_type, npy_direct_func_proc);
    }
}
