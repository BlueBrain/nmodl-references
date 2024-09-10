/*********************************************************
Model Name      : hodhux
Filename        : hodhux.mod
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

static constexpr auto number_of_datum_variables = 6;
static constexpr auto number_of_floating_point_variables = 23;

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
extern double celsius;


namespace neuron {
    #ifndef NRN_PRCELLSTATE
    #define NRN_PRCELLSTATE 0
    #endif


    /** channel information */
    static const char *mechanism_info[] = {
        "7.7.0",
        "hodhux",
        "gnabar_hodhux",
        "gkbar_hodhux",
        "gl_hodhux",
        "el_hodhux",
        0,
        "il_hodhux",
        "minf_hodhux",
        "hinf_hodhux",
        "ninf_hodhux",
        "mexp_hodhux",
        "hexp_hodhux",
        "nexp_hodhux",
        0,
        "m_hodhux",
        "h_hodhux",
        "n_hodhux",
        0,
        0
    };


    /* NEURON global variables */
    static Symbol* _na_sym;
    static Symbol* _k_sym;
    static int mech_type;
    static Prop* _extcall_prop;
    /* _prop_id kind of shadows _extcall_prop to allow validity checking. */
    static _nrn_non_owning_id_without_container _prop_id{};
    static int hoc_nrnpointerindex = -1;
    static _nrn_mechanism_std_vector<Datum> _extcall_thread;


    /** all global variables */
    struct hodhux_Store {
        double m0{0};
        double h0{0};
        double n0{0};
    };
    static_assert(std::is_trivially_copy_constructible_v<hodhux_Store>);
    static_assert(std::is_trivially_move_constructible_v<hodhux_Store>);
    static_assert(std::is_trivially_copy_assignable_v<hodhux_Store>);
    static_assert(std::is_trivially_move_assignable_v<hodhux_Store>);
    static_assert(std::is_trivially_destructible_v<hodhux_Store>);
    hodhux_Store hodhux_global;
    static std::vector<double> _parameter_defaults = {
        0.12 /* gnabar */,
        0.036 /* gkbar */,
        0.0003 /* gl */,
        -54.3 /* el */
    };


    /** all mechanism instance variables and global variables */
    struct hodhux_Instance  {
        double* celsius{&::celsius};
        double* gnabar{};
        double* gkbar{};
        double* gl{};
        double* el{};
        double* il{};
        double* minf{};
        double* hinf{};
        double* ninf{};
        double* mexp{};
        double* hexp{};
        double* nexp{};
        double* m{};
        double* h{};
        double* n{};
        double* ena{};
        double* ek{};
        double* Dm{};
        double* Dh{};
        double* Dn{};
        double* ina{};
        double* ik{};
        double* v_unused{};
        double* g_unused{};
        const double* const* ion_ena{};
        double* const* ion_ina{};
        double* const* ion_dinadv{};
        const double* const* ion_ek{};
        double* const* ion_ik{};
        double* const* ion_dikdv{};
        hodhux_Store* global{&hodhux_global};
    };


    struct hodhux_NodeData  {
        int const * nodeindices;
        double const * node_voltages;
        double * node_diagonal;
        double * node_rhs;
        int nodecount;
    };


    static hodhux_Instance make_instance_hodhux(_nrn_mechanism_cache_range& _lmc) {
        return hodhux_Instance {
            &::celsius,
            _lmc.template fpfield_ptr<0>(),
            _lmc.template fpfield_ptr<1>(),
            _lmc.template fpfield_ptr<2>(),
            _lmc.template fpfield_ptr<3>(),
            _lmc.template fpfield_ptr<4>(),
            _lmc.template fpfield_ptr<5>(),
            _lmc.template fpfield_ptr<6>(),
            _lmc.template fpfield_ptr<7>(),
            _lmc.template fpfield_ptr<8>(),
            _lmc.template fpfield_ptr<9>(),
            _lmc.template fpfield_ptr<10>(),
            _lmc.template fpfield_ptr<11>(),
            _lmc.template fpfield_ptr<12>(),
            _lmc.template fpfield_ptr<13>(),
            _lmc.template fpfield_ptr<14>(),
            _lmc.template fpfield_ptr<15>(),
            _lmc.template fpfield_ptr<16>(),
            _lmc.template fpfield_ptr<17>(),
            _lmc.template fpfield_ptr<18>(),
            _lmc.template fpfield_ptr<19>(),
            _lmc.template fpfield_ptr<20>(),
            _lmc.template fpfield_ptr<21>(),
            _lmc.template fpfield_ptr<22>(),
            _lmc.template dptr_field_ptr<0>(),
            _lmc.template dptr_field_ptr<1>(),
            _lmc.template dptr_field_ptr<2>(),
            _lmc.template dptr_field_ptr<3>(),
            _lmc.template dptr_field_ptr<4>(),
            _lmc.template dptr_field_ptr<5>()
        };
    }


    static hodhux_NodeData make_node_data_hodhux(NrnThread& nt, Memb_list& _ml_arg) {
        return hodhux_NodeData {
            _ml_arg.nodeindices,
            nt.node_voltage_storage(),
            nt.node_d_storage(),
            nt.node_rhs_storage(),
            _ml_arg.nodecount
        };
    }
    static hodhux_NodeData make_node_data_hodhux(Prop * _prop) {
        static std::vector<int> node_index{0};
        Node* _node = _nrn_mechanism_access_node(_prop);
        return hodhux_NodeData {
            node_index.data(),
            &_nrn_mechanism_access_voltage(_node),
            &_nrn_mechanism_access_d(_node),
            &_nrn_mechanism_access_rhs(_node),
            1
        };
    }

    void nrn_destructor_hodhux(Prop* prop) {
        Datum* _ppvar = _nrn_mechanism_access_dparam(prop);
        _nrn_mechanism_cache_instance _lmc{prop};
        const size_t id = 0;
        auto inst = make_instance_hodhux(_lmc);
        auto node_data = make_node_data_hodhux(prop);

    }


    static void nrn_alloc_hodhux(Prop* _prop) {
        Datum *_ppvar = nullptr;
        _ppvar = nrn_prop_datum_alloc(mech_type, 6, _prop);
        _nrn_mechanism_access_dparam(_prop) = _ppvar;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const _iml = 0;
        assert(_nrn_mechanism_get_num_vars(_prop) == 23);
        /*initialize range parameters*/
        _lmc.template fpfield<0>(_iml) = _parameter_defaults[0]; /* gnabar */
        _lmc.template fpfield<1>(_iml) = _parameter_defaults[1]; /* gkbar */
        _lmc.template fpfield<2>(_iml) = _parameter_defaults[2]; /* gl */
        _lmc.template fpfield<3>(_iml) = _parameter_defaults[3]; /* el */
        _nrn_mechanism_access_dparam(_prop) = _ppvar;
        Symbol * na_sym = hoc_lookup("na_ion");
        Prop * na_prop = need_memb(na_sym);
        nrn_promote(na_prop, 0, 1);
        _ppvar[0] = _nrn_mechanism_get_param_handle(na_prop, 0);
        _ppvar[1] = _nrn_mechanism_get_param_handle(na_prop, 3);
        _ppvar[2] = _nrn_mechanism_get_param_handle(na_prop, 4);
        Symbol * k_sym = hoc_lookup("k_ion");
        Prop * k_prop = need_memb(k_sym);
        nrn_promote(k_prop, 0, 1);
        _ppvar[3] = _nrn_mechanism_get_param_handle(k_prop, 0);
        _ppvar[4] = _nrn_mechanism_get_param_handle(k_prop, 3);
        _ppvar[5] = _nrn_mechanism_get_param_handle(k_prop, 4);
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
    inline double vtrap_hodhux(_nrn_mechanism_cache_range& _lmc, hodhux_Instance& inst, hodhux_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _lx, double _ly);
    inline int states_hodhux(_nrn_mechanism_cache_range& _lmc, hodhux_Instance& inst, hodhux_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt);
    inline int rates_hodhux(_nrn_mechanism_cache_range& _lmc, hodhux_Instance& inst, hodhux_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _lv);


    /** connect global (scalar) variables to hoc -- */
    static DoubScal hoc_scalar_double[] = {
        {nullptr, nullptr}
    };


    /** connect global (array) variables to hoc -- */
    static DoubVec hoc_vector_double[] = {
        {nullptr, nullptr, 0}
    };


    /* declaration of user functions */
    static void _hoc_states(void);
    static void _hoc_rates(void);
    static void _hoc_vtrap(void);
    static double _npy_states(Prop*);
    static double _npy_rates(Prop*);
    static double _npy_vtrap(Prop*);


    /* connect user functions to hoc names */
    static VoidFunc hoc_intfunc[] = {
        {"setdata_hodhux", _hoc_setdata},
        {"states_hodhux", _hoc_states},
        {"rates_hodhux", _hoc_rates},
        {"vtrap_hodhux", _hoc_vtrap},
        {nullptr, nullptr}
    };
    static NPyDirectMechFunc npy_direct_func_proc[] = {
        {"states", _npy_states},
        {"rates", _npy_rates},
        {"vtrap", _npy_vtrap},
        {nullptr, nullptr}
    };
    static void _hoc_states(void) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        if (!_prop_id) {
            hoc_execerror("No data for states_hodhux. Requires prior call to setdata_hodhux and that the specified mechanism instance still be in existence.", NULL);
        }
        Prop* _local_prop = _extcall_prop;
        _nrn_mechanism_cache_instance _lmc{_local_prop};
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_hodhux(_lmc);
        auto node_data = make_node_data_hodhux(_local_prop);
        _r = 1.;
        states_hodhux(_lmc, inst, node_data, id, _ppvar, _thread, nt);
        hoc_retpushx(_r);
    }
    static double _npy_states(Prop* _prop) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id = 0;
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_hodhux(_lmc);
        auto node_data = make_node_data_hodhux(_prop);
        _r = 1.;
        states_hodhux(_lmc, inst, node_data, id, _ppvar, _thread, nt);
        return(_r);
    }
    static void _hoc_rates(void) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        if (!_prop_id) {
            hoc_execerror("No data for rates_hodhux. Requires prior call to setdata_hodhux and that the specified mechanism instance still be in existence.", NULL);
        }
        Prop* _local_prop = _extcall_prop;
        _nrn_mechanism_cache_instance _lmc{_local_prop};
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_hodhux(_lmc);
        auto node_data = make_node_data_hodhux(_local_prop);
        _r = 1.;
        rates_hodhux(_lmc, inst, node_data, id, _ppvar, _thread, nt, *getarg(1));
        hoc_retpushx(_r);
    }
    static double _npy_rates(Prop* _prop) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id = 0;
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_hodhux(_lmc);
        auto node_data = make_node_data_hodhux(_prop);
        _r = 1.;
        rates_hodhux(_lmc, inst, node_data, id, _ppvar, _thread, nt, *getarg(1));
        return(_r);
    }
    static void _hoc_vtrap(void) {
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
        auto inst = make_instance_hodhux(_lmc);
        auto node_data = make_node_data_hodhux(_local_prop);
        _r = vtrap_hodhux(_lmc, inst, node_data, id, _ppvar, _thread, nt, *getarg(1), *getarg(2));
        hoc_retpushx(_r);
    }
    static double _npy_vtrap(Prop* _prop) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id = 0;
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_hodhux(_lmc);
        auto node_data = make_node_data_hodhux(_prop);
        _r = vtrap_hodhux(_lmc, inst, node_data, id, _ppvar, _thread, nt, *getarg(1), *getarg(2));
        return(_r);
    }


    inline int states_hodhux(_nrn_mechanism_cache_range& _lmc, hodhux_Instance& inst, hodhux_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt) {
        int ret_states = 0;
        auto v = node_data.node_voltages[node_data.nodeindices[id]];
        rates_hodhux(_lmc, inst, node_data, id, _ppvar, _thread, nt, v);
        inst.m[id] = inst.m[id] + inst.mexp[id] * (inst.minf[id] - inst.m[id]);
        inst.h[id] = inst.h[id] + inst.hexp[id] * (inst.hinf[id] - inst.h[id]);
        inst.n[id] = inst.n[id] + inst.nexp[id] * (inst.ninf[id] - inst.n[id]);
        return ret_states;
    }


    inline int rates_hodhux(_nrn_mechanism_cache_range& _lmc, hodhux_Instance& inst, hodhux_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _lv) {
        int ret_rates = 0;
        auto v = node_data.node_voltages[node_data.nodeindices[id]];
        double q10, tinc, alpha, beta, sum;
        q10 = pow(3.0, ((*(inst.celsius) - 6.3) / 10.0));
        tinc =  -nt->_dt * q10;
        alpha = .1 * vtrap_hodhux(_lmc, inst, node_data, id, _ppvar, _thread, nt,  -(_lv + 40.0), 10.0);
        beta = 4.0 * exp( -(_lv + 65.0) / 18.0);
        sum = alpha + beta;
        inst.minf[id] = alpha / sum;
        inst.mexp[id] = 1.0 - exp(tinc * sum);
        alpha = .07 * exp( -(_lv + 65.0) / 20.0);
        beta = 1.0 / (exp( -(_lv + 35.0) / 10.0) + 1.0);
        sum = alpha + beta;
        inst.hinf[id] = alpha / sum;
        inst.hexp[id] = 1.0 - exp(tinc * sum);
        alpha = .01 * vtrap_hodhux(_lmc, inst, node_data, id, _ppvar, _thread, nt,  -(_lv + 55.0), 10.0);
        beta = .125 * exp( -(_lv + 65.0) / 80.0);
        sum = alpha + beta;
        inst.ninf[id] = alpha / sum;
        inst.nexp[id] = 1.0 - exp(tinc * sum);
        return ret_rates;
    }


    inline double vtrap_hodhux(_nrn_mechanism_cache_range& _lmc, hodhux_Instance& inst, hodhux_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _lx, double _ly) {
        double ret_vtrap = 0.0;
        auto v = node_data.node_voltages[node_data.nodeindices[id]];
        if (fabs(_lx / _ly) < 1e-6) {
            ret_vtrap = _ly * (1.0 - _lx / _ly / 2.0);
        } else {
            ret_vtrap = _lx / (exp(_lx / _ly) - 1.0);
        }
        return ret_vtrap;
    }


    void nrn_init_hodhux(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _type};
        auto inst = make_instance_hodhux(_lmc);
        auto node_data = make_node_data_hodhux(*nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* _thread = _ml_arg->_thread;
        for (int id = 0; id < nodecount; id++) {
            auto* _ppvar = _ml_arg->pdata[id];
            int node_id = node_data.nodeindices[id];
            auto v = node_data.node_voltages[node_id];
            inst.m[id] = inst.global->m0;
            inst.h[id] = inst.global->h0;
            inst.n[id] = inst.global->n0;
            inst.ena[id] = (*inst.ion_ena[id]);
            inst.ek[id] = (*inst.ion_ek[id]);
            rates_hodhux(_lmc, inst, node_data, id, _ppvar, _thread, nt, v);
            inst.m[id] = inst.minf[id];
            inst.h[id] = inst.hinf[id];
            inst.n[id] = inst.ninf[id];
        }
    }


    inline double nrn_current_hodhux(_nrn_mechanism_cache_range& _lmc, NrnThread* nt, Datum* _ppvar, Datum* _thread, size_t id, hodhux_Instance& inst, hodhux_NodeData& node_data, double v) {
        double current = 0.0;
        inst.ina[id] = inst.gnabar[id] * inst.m[id] * inst.m[id] * inst.m[id] * inst.h[id] * (v - inst.ena[id]);
        inst.ik[id] = inst.gkbar[id] * inst.n[id] * inst.n[id] * inst.n[id] * inst.n[id] * (v - inst.ek[id]);
        inst.il[id] = inst.gl[id] * (v - inst.el[id]);
        current += inst.il[id];
        current += inst.ina[id];
        current += inst.ik[id];
        return current;
    }


    /** update current */
    void nrn_cur_hodhux(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _type};
        auto inst = make_instance_hodhux(_lmc);
        auto node_data = make_node_data_hodhux(*nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* _thread = _ml_arg->_thread;
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_data.nodeindices[id];
            double v = node_data.node_voltages[node_id];
            auto* _ppvar = _ml_arg->pdata[id];
            inst.ena[id] = (*inst.ion_ena[id]);
            inst.ek[id] = (*inst.ion_ek[id]);
            double I1 = nrn_current_hodhux(_lmc, nt, _ppvar, _thread, id, inst, node_data, v+0.001);
            double dina = inst.ina[id];
            double dik = inst.ik[id];
            double I0 = nrn_current_hodhux(_lmc, nt, _ppvar, _thread, id, inst, node_data, v);
            double rhs = I0;
            double g = (I1-I0)/0.001;
            (*inst.ion_dinadv[id]) += (dina-inst.ina[id])/0.001;
            (*inst.ion_dikdv[id]) += (dik-inst.ik[id])/0.001;
            (*inst.ion_ina[id]) += inst.ina[id];
            (*inst.ion_ik[id]) += inst.ik[id];
            node_data.node_rhs[node_id] -= rhs;
            inst.g_unused[id] = g;
        }
    }


    void nrn_state_hodhux(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _type};
        auto inst = make_instance_hodhux(_lmc);
        auto node_data = make_node_data_hodhux(*nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* _thread = _ml_arg->_thread;
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_data.nodeindices[id];
            auto* _ppvar = _ml_arg->pdata[id];
            auto v = node_data.node_voltages[node_id];
            inst.ena[id] = (*inst.ion_ena[id]);
            inst.ek[id] = (*inst.ion_ek[id]);
            rates_hodhux(_lmc, inst, node_data, id, _ppvar, _thread, nt, v);
            inst.m[id] = inst.m[id] + inst.mexp[id] * (inst.minf[id] - inst.m[id]);
            inst.h[id] = inst.h[id] + inst.hexp[id] * (inst.hinf[id] - inst.h[id]);
            inst.n[id] = inst.n[id] + inst.nexp[id] * (inst.ninf[id] - inst.n[id]);
        }
    }


    static void nrn_jacob_hodhux(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _type};
        auto inst = make_instance_hodhux(_lmc);
        auto node_data = make_node_data_hodhux(*nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_data.nodeindices[id];
            node_data.node_diagonal[node_id] += inst.g_unused[id];
        }
    }


    static void _initlists() {
    }


    /** register channel with the simulator */
    extern "C" void _hodhux_reg() {
        _initlists();

        ion_reg("na", -10000);
        ion_reg("k", -10000);

        _na_sym = hoc_lookup("na_ion");
        _k_sym = hoc_lookup("k_ion");

        register_mech(mechanism_info, nrn_alloc_hodhux, nrn_cur_hodhux, nrn_jacob_hodhux, nrn_state_hodhux, nrn_init_hodhux, hoc_nrnpointerindex, 1);

        mech_type = nrn_get_mechtype(mechanism_info[1]);
        hoc_register_parm_default(mech_type, &_parameter_defaults);
        _nrn_mechanism_register_data_fields(mech_type,
            _nrn_mechanism_field<double>{"gnabar"} /* 0 */,
            _nrn_mechanism_field<double>{"gkbar"} /* 1 */,
            _nrn_mechanism_field<double>{"gl"} /* 2 */,
            _nrn_mechanism_field<double>{"el"} /* 3 */,
            _nrn_mechanism_field<double>{"il"} /* 4 */,
            _nrn_mechanism_field<double>{"minf"} /* 5 */,
            _nrn_mechanism_field<double>{"hinf"} /* 6 */,
            _nrn_mechanism_field<double>{"ninf"} /* 7 */,
            _nrn_mechanism_field<double>{"mexp"} /* 8 */,
            _nrn_mechanism_field<double>{"hexp"} /* 9 */,
            _nrn_mechanism_field<double>{"nexp"} /* 10 */,
            _nrn_mechanism_field<double>{"m"} /* 11 */,
            _nrn_mechanism_field<double>{"h"} /* 12 */,
            _nrn_mechanism_field<double>{"n"} /* 13 */,
            _nrn_mechanism_field<double>{"ena"} /* 14 */,
            _nrn_mechanism_field<double>{"ek"} /* 15 */,
            _nrn_mechanism_field<double>{"Dm"} /* 16 */,
            _nrn_mechanism_field<double>{"Dh"} /* 17 */,
            _nrn_mechanism_field<double>{"Dn"} /* 18 */,
            _nrn_mechanism_field<double>{"ina"} /* 19 */,
            _nrn_mechanism_field<double>{"ik"} /* 20 */,
            _nrn_mechanism_field<double>{"v_unused"} /* 21 */,
            _nrn_mechanism_field<double>{"g_unused"} /* 22 */,
            _nrn_mechanism_field<double*>{"ion_ena", "na_ion"} /* 0 */,
            _nrn_mechanism_field<double*>{"ion_ina", "na_ion"} /* 1 */,
            _nrn_mechanism_field<double*>{"ion_dinadv", "na_ion"} /* 2 */,
            _nrn_mechanism_field<double*>{"ion_ek", "k_ion"} /* 3 */,
            _nrn_mechanism_field<double*>{"ion_ik", "k_ion"} /* 4 */,
            _nrn_mechanism_field<double*>{"ion_dikdv", "k_ion"} /* 5 */
        );

        hoc_register_prop_size(mech_type, 23, 6);
        hoc_register_dparam_semantics(mech_type, 0, "na_ion");
        hoc_register_dparam_semantics(mech_type, 1, "na_ion");
        hoc_register_dparam_semantics(mech_type, 2, "na_ion");
        hoc_register_dparam_semantics(mech_type, 3, "k_ion");
        hoc_register_dparam_semantics(mech_type, 4, "k_ion");
        hoc_register_dparam_semantics(mech_type, 5, "k_ion");
        hoc_register_var(hoc_scalar_double, hoc_vector_double, hoc_intfunc);
        hoc_register_npy_direct(mech_type, npy_direct_func_proc);
    }
}
