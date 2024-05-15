/*********************************************************
Model Name      : tbl
Filename        : table.mod
NMODL Version   : 7.7.0
Vectorized      : true
Threadsafe      : true
Created         : DATE
Simulator       : NEURON
Backend         : C++ (api-compatibility)
NMODL Compiler  : VERSION
*********************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "mech_api.h"
#include "neuron/cache/mechanism_range.hpp"
#include "nrniv_mf.h"
#include "section_fwd.hpp"

/* NEURON global macro definitions */
/* VECTORIZED */
#define NRN_VECTORIZED 1

static constexpr auto number_of_datum_variables = 0;
static constexpr auto number_of_floating_point_variables = 7;

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


namespace neuron {
    #ifndef NRN_PRCELLSTATE
    #define NRN_PRCELLSTATE 0
    #endif


    /** channel information */
    static const char *mechanism_info[] = {
        "7.7.0",
        "tbl",
        "e_tbl",
        "gmax_tbl",
        0,
        "g_tbl",
        "i_tbl",
        0,
        0,
        0
    };


    /* NEURON global variables */
    static int mech_type;
    static Prop* _extcall_prop;
    /* _prop_id kind of shadows _extcall_prop to allow validity checking. */
    static _nrn_non_owning_id_without_container _prop_id{};
    static int hoc_nrnpointerindex = -1;
    static _nrn_mechanism_std_vector<Datum> _extcall_thread;


    /** all global variables */
    struct tbl_Store {
        double usetable{1};
        double tmin_sigmoid1{};
        double mfac_sigmoid1{};
        double t_sig[101]{};
        double k{0.1};
        double d{-50};
    };
    static_assert(std::is_trivially_copy_constructible_v<tbl_Store>);
    static_assert(std::is_trivially_move_constructible_v<tbl_Store>);
    static_assert(std::is_trivially_copy_assignable_v<tbl_Store>);
    static_assert(std::is_trivially_move_assignable_v<tbl_Store>);
    static_assert(std::is_trivially_destructible_v<tbl_Store>);
    tbl_Store tbl_global;


    /** all mechanism instance variables and global variables */
    struct tbl_Instance  {
        double* e{};
        double* gmax{};
        double* g{};
        double* i{};
        double* sig{};
        double* v_unused{};
        double* g_unused{};
        tbl_Store* global{&tbl_global};
    };


    struct tbl_NodeData  {
        int const * nodeindices;
        double const * node_voltages;
        double * node_diagonal;
        double * node_rhs;
        int nodecount;
    };


    static tbl_Instance make_instance_tbl(_nrn_mechanism_cache_range& _ml) {
        return tbl_Instance {
            _ml.template fpfield_ptr<0>(),
            _ml.template fpfield_ptr<1>(),
            _ml.template fpfield_ptr<2>(),
            _ml.template fpfield_ptr<3>(),
            _ml.template fpfield_ptr<4>(),
            _ml.template fpfield_ptr<5>(),
            _ml.template fpfield_ptr<6>()
        };
    }


    static tbl_NodeData make_node_data_tbl(NrnThread& _nt, Memb_list& _ml_arg) {
        return tbl_NodeData {
            _ml_arg.nodeindices,
            _nt.node_voltage_storage(),
            _nt.node_d_storage(),
            _nt.node_rhs_storage(),
            _ml_arg.nodecount
        };
    }


    static void nrn_alloc_tbl(Prop* _prop) {
        Prop *prop_ion{};
        Datum *_ppvar{};
        _nrn_mechanism_cache_instance _ml_real{_prop};
        auto* const _ml = &_ml_real;
        size_t const _iml{};
        assert(_nrn_mechanism_get_num_vars(_prop) == 7);
        /*initialize range parameters*/
        _ml->template fpfield<0>(_iml) = 0; /* e */
        _ml->template fpfield<1>(_iml) = 0; /* gmax */
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
    inline int sigmoid1_tbl(_nrn_mechanism_cache_range* _ml, tbl_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double v);
    void check_sigmoid1_tbl(_nrn_mechanism_cache_range* _ml, tbl_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* _nt);
    static void _check_table_thread(Memb_list* _ml, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* _nt, int _type, _nrn_model_sorted_token const& _sorted_token)
{
        _nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml, _type};
        auto inst = make_instance_tbl(_lmr);
        check_sigmoid1_tbl(&_lmr, inst, id, _ppvar, _thread, _nt);
    }


    /** connect global (scalar) variables to hoc -- */
    static DoubScal hoc_scalar_double[] = {
        {"k_tbl", &tbl_global.k},
        {"d_tbl", &tbl_global.d},
        {"usetable_tbl", &tbl_global.usetable},
        {nullptr, nullptr}
    };


    /** connect global (array) variables to hoc -- */
    static DoubVec hoc_vector_double[] = {
        {nullptr, nullptr, 0}
    };


    /* declaration of user functions */
    static void _hoc_sigmoid1(void);
    static double _npy_sigmoid1(Prop*);


    /* connect user functions to hoc names */
    static VoidFunc hoc_intfunc[] = {
        {"setdata_tbl", _hoc_setdata},
        {"sigmoid1_tbl", _hoc_sigmoid1},
        {0, 0}
    };
    static NPyDirectMechFunc npy_direct_func_proc[] = {
        {"sigmoid1", _npy_sigmoid1},
    };
    static void _hoc_sigmoid1(void) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* _nt;
        Prop* _local_prop = _prop_id ? _extcall_prop : nullptr;
        _nrn_mechanism_cache_instance _ml_real{_local_prop};
        auto* const _ml = &_ml_real;
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        _nt = nrn_threads;
        auto inst = make_instance_tbl(_ml_real);
        check_sigmoid1_tbl(_ml, inst, id, _ppvar, _thread, _nt);
        _r = 1.;
        sigmoid1_tbl(_ml, inst, id, _ppvar, _thread, _nt, *getarg(1));
        hoc_retpushx(_r);
    }
    static double _npy_sigmoid1(Prop* _prop) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* _nt;
        _nrn_mechanism_cache_instance _ml_real{_prop};
        auto* const _ml = &_ml_real;
        size_t const id{};
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        _nt = nrn_threads;
        auto inst = make_instance_tbl(_ml_real);
        check_sigmoid1_tbl(_ml, inst, id, _ppvar, _thread, _nt);
        _r = 1.;
        sigmoid1_tbl(_ml, inst, id, _ppvar, _thread, _nt, *getarg(1));
        return(_r);
    }


    inline int f_sigmoid1_tbl(_nrn_mechanism_cache_range* _ml, tbl_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double v) {
        int ret_f_sigmoid1 = 0;
        inst.sig[id] = 1.0 / (1.0 + exp(inst.global->k * (v - inst.global->d)));
        return ret_f_sigmoid1;
    }


    void check_sigmoid1_tbl(_nrn_mechanism_cache_range* _ml, tbl_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
        if (inst.global->usetable == 0) {
            return;
        }
        static bool make_table = true;
        static double save_k;
        static double save_d;
        if (save_k != inst.global->k) {
            make_table = true;
        }
        if (save_d != inst.global->d) {
            make_table = true;
        }
        if (make_table) {
            make_table = false;
            inst.global->tmin_sigmoid1 =  -100.0;
            double tmax = 100.0;
            double dx = (tmax-inst.global->tmin_sigmoid1) / 100.;
            inst.global->mfac_sigmoid1 = 1./dx;
            double x = inst.global->tmin_sigmoid1;
            for (std::size_t i = 0; i < 101; x += dx, i++) {
                f_sigmoid1_tbl(_ml, inst, id, _ppvar, _thread, _nt, x);
                inst.global->t_sig[i] = inst.sig[id];
            }
            save_k = inst.global->k;
            save_d = inst.global->d;
        }
    }


    inline int sigmoid1_tbl(_nrn_mechanism_cache_range* _ml, tbl_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double v){
        if (inst.global->usetable == 0) {
            f_sigmoid1_tbl(_ml, inst, id, _ppvar, _thread, _nt, v);
            return 0;
        }
        double xi = inst.global->mfac_sigmoid1 * (v - inst.global->tmin_sigmoid1);
        if (isnan(xi)) {
            inst.sig[id] = xi;
            return 0;
        }
        if (xi <= 0. || xi >= 100.) {
            int index = (xi <= 0.) ? 0 : 100;
            inst.sig[id] = inst.global->t_sig[index];
            return 0;
        }
        int i = int(xi);
        double theta = xi - double(i);
        inst.sig[id] = inst.global->t_sig[i] + theta*(inst.global->t_sig[i+1]-inst.global->t_sig[i]);
        return 0;
    }


    void nrn_init_tbl(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
        auto inst = make_instance_tbl(_lmr);
        auto node_data = make_node_data_tbl(*_nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* const _ml = &_lmr;
        auto* _thread = _ml_arg->_thread;
        for (int id = 0; id < nodecount; id++) {
            
            int node_id = node_data.nodeindices[id];
            auto* _ppvar = _ml_arg->pdata[id];
            auto v = node_data.node_voltages[node_id];
            inst.v_unused[id] = v;
        }
    }


    inline double nrn_current_tbl(_nrn_mechanism_cache_range* _ml, NrnThread* _nt, Datum* _ppvar, Datum* _thread, size_t id, tbl_Instance& inst, tbl_NodeData& node_data, double v) {
        double current = 0.0;
        sigmoid1_tbl(_ml, inst, id, _ppvar, _thread, _nt, v);
        inst.g[id] = inst.gmax[id] * inst.sig[id];
        inst.i[id] = inst.g[id] * (v - inst.e[id]);
        current += inst.i[id];
        return current;
    }


    /** update current */
    void nrn_cur_tbl(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
        auto inst = make_instance_tbl(_lmr);
        auto node_data = make_node_data_tbl(*_nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* const _ml = &_lmr;
        auto* _thread = _ml_arg->_thread;
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_data.nodeindices[id];
            double v = node_data.node_voltages[node_id];
            auto* _ppvar = _ml_arg->pdata[id];
            double I1 = nrn_current_tbl(_ml, _nt, _ppvar, _thread, id, inst, node_data, v+0.001);
            double I0 = nrn_current_tbl(_ml, _nt, _ppvar, _thread, id, inst, node_data, v);
            double rhs = I0;
            double g = (I1-I0)/0.001;
            node_data.node_rhs[node_id] -= rhs;
            // remember the conductances so we can set them later
            inst.g_unused[id] = g;
        }
    }


    void nrn_state_tbl(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
        auto inst = make_instance_tbl(_lmr);
        auto node_data = make_node_data_tbl(*_nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* const _ml = &_lmr;
        auto* _thread = _ml_arg->_thread;
        for (int id = 0; id < nodecount; id++) {
            
            int node_id = node_data.nodeindices[id];
            auto* _ppvar = _ml_arg->pdata[id];
            auto v = node_data.node_voltages[node_id];
        }
    }


    /** nrn_jacob function */
    static void nrn_jacob_tbl(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
        auto inst = make_instance_tbl(_lmr);
        auto node_data = make_node_data_tbl(*_nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
            // set conductances properly
            int node_id = node_data.nodeindices[id];
            node_data.node_diagonal[node_id] += inst.g_unused[id];
        }
    }


    static void _initlists() {
    }


    /** register channel with the simulator */
    extern "C" void _table_reg() {
        _initlists();



        register_mech(mechanism_info, nrn_alloc_tbl, nrn_cur_tbl, nrn_jacob_tbl, nrn_state_tbl, nrn_init_tbl, hoc_nrnpointerindex, 1);

        mech_type = nrn_get_mechtype(mechanism_info[1]);
        _nrn_thread_table_reg(mech_type, _check_table_thread);
        _nrn_mechanism_register_data_fields(mech_type,
            _nrn_mechanism_field<double>{"e"} /* 0 */,
            _nrn_mechanism_field<double>{"gmax"} /* 1 */,
            _nrn_mechanism_field<double>{"g"} /* 2 */,
            _nrn_mechanism_field<double>{"i"} /* 3 */,
            _nrn_mechanism_field<double>{"sig"} /* 4 */,
            _nrn_mechanism_field<double>{"v_unused"} /* 5 */,
            _nrn_mechanism_field<double>{"g_unused"} /* 6 */
        );

        hoc_register_prop_size(mech_type, 7, 0);
        hoc_register_var(hoc_scalar_double, hoc_vector_double, hoc_intfunc);
        hoc_register_npy_direct(mech_type, npy_direct_func_proc);
    }
}
