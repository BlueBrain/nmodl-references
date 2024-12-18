/*********************************************************
Model Name      : art_function_table
Filename        : art_function_table.mod
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

static constexpr auto number_of_datum_variables = 2;
static constexpr auto number_of_floating_point_variables = 1;

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

extern Prop* nrn_point_prop_;
extern void _cvode_abstol(Symbol**, double*, int);
extern Node* nrn_alloc_node_;


namespace neuron {
    #ifndef NRN_PRCELLSTATE
    #define NRN_PRCELLSTATE 0
    #endif


    /** channel information */
    static const char *mechanism_info[] = {
        "7.7.0",
        "art_function_table",
        0,
        0,
        0,
        0
    };


    /* NEURON global variables */
    static int mech_type;
    static int _pointtype;
    static _nrn_mechanism_std_vector<Datum> _extcall_thread;


    /** all global variables */
    struct art_function_table_Store {
        void* _ptable_cnst1{};
        void* _ptable_cnst2{};
        void* _ptable_tau1{};
        void* _ptable_tau2{};
    };
    static_assert(std::is_trivially_copy_constructible_v<art_function_table_Store>);
    static_assert(std::is_trivially_move_constructible_v<art_function_table_Store>);
    static_assert(std::is_trivially_copy_assignable_v<art_function_table_Store>);
    static_assert(std::is_trivially_move_assignable_v<art_function_table_Store>);
    static_assert(std::is_trivially_destructible_v<art_function_table_Store>);
    static art_function_table_Store art_function_table_global;
    auto _ptable_cnst1_art_function_table() -> std::decay<decltype(art_function_table_global._ptable_cnst1)>::type  {
        return art_function_table_global._ptable_cnst1;
    }
    auto _ptable_cnst2_art_function_table() -> std::decay<decltype(art_function_table_global._ptable_cnst2)>::type  {
        return art_function_table_global._ptable_cnst2;
    }
    auto _ptable_tau1_art_function_table() -> std::decay<decltype(art_function_table_global._ptable_tau1)>::type  {
        return art_function_table_global._ptable_tau1;
    }
    auto _ptable_tau2_art_function_table() -> std::decay<decltype(art_function_table_global._ptable_tau2)>::type  {
        return art_function_table_global._ptable_tau2;
    }

    static std::vector<double> _parameter_defaults = {
    };


    /** all mechanism instance variables and global variables */
    struct art_function_table_Instance  {
        double* v_unused{};
        const double* const* node_area{};
        art_function_table_Store* global{&art_function_table_global};
    };


    struct art_function_table_NodeData  {
        int const * nodeindices;
        double const * node_voltages;
        double * node_diagonal;
        double * node_rhs;
        int nodecount;
    };


    static art_function_table_Instance make_instance_art_function_table(_nrn_mechanism_cache_range* _lmc) {
        if(_lmc == nullptr) {
            return art_function_table_Instance();
        }

        return art_function_table_Instance {
            _lmc->template fpfield_ptr<0>(),
            _lmc->template dptr_field_ptr<0>()
        };
    }


    static art_function_table_NodeData make_node_data_art_function_table(NrnThread& nt, Memb_list& _ml_arg) {
        return art_function_table_NodeData {
            _ml_arg.nodeindices,
            nt.node_voltage_storage(),
            nt.node_d_storage(),
            nt.node_rhs_storage(),
            _ml_arg.nodecount
        };
    }
    static art_function_table_NodeData make_node_data_art_function_table(Prop * _prop) {
        if(!_prop) {
            return art_function_table_NodeData();
        }

        static std::vector<int> node_index{0};
        Node* _node = _nrn_mechanism_access_node(_prop);
        return art_function_table_NodeData {
            node_index.data(),
            &_nrn_mechanism_access_voltage(_node),
            &_nrn_mechanism_access_d(_node),
            &_nrn_mechanism_access_rhs(_node),
            1
        };
    }

    static void nrn_destructor_art_function_table(Prop* prop);


    static void nrn_alloc_art_function_table(Prop* _prop) {
        Datum *_ppvar = nullptr;
        if (nrn_point_prop_) {
            _nrn_mechanism_access_alloc_seq(_prop) = _nrn_mechanism_access_alloc_seq(nrn_point_prop_);
            _ppvar = _nrn_mechanism_access_dparam(nrn_point_prop_);
        } else {
            _ppvar = nrn_prop_datum_alloc(mech_type, 2, _prop);
            _nrn_mechanism_access_dparam(_prop) = _ppvar;
            _nrn_mechanism_cache_instance _lmc{_prop};
            size_t const _iml = 0;
            assert(_nrn_mechanism_get_num_vars(_prop) == 1);
            /*initialize range parameters*/
        }
        _nrn_mechanism_access_dparam(_prop) = _ppvar;
        if(!nrn_point_prop_) {
        }
    }


    /* Mechanism procedures and functions */
    inline static double use_tau2_art_function_table(_nrn_mechanism_cache_range& _lmc, art_function_table_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _lv, double _lx);
    double cnst1_art_function_table(double v);
    double table_cnst1_art_function_table();
    double cnst2_art_function_table(double v, double x);
    double table_cnst2_art_function_table();
    double tau1_art_function_table(double v);
    double table_tau1_art_function_table();
    double tau2_art_function_table(double v, double x);
    double table_tau2_art_function_table();
    static void _apply_diffusion_function(ldifusfunc2_t _f, const _nrn_model_sorted_token& _sorted_token, NrnThread& _nt) {
    }

    /* Point Process specific functions */
    static void* _hoc_create_pnt(Object* _ho) {
        return create_point_process(_pointtype, _ho);
    }
    static void _hoc_destroy_pnt(void* _vptr) {
        destroy_point_process(_vptr);
    }
    static double _hoc_loc_pnt(void* _vptr) {
        return loc_point_process(_pointtype, _vptr);
    }
    static double _hoc_has_loc(void* _vptr) {
        return has_loc_point(_vptr);
    }
    static double _hoc_get_loc_pnt(void* _vptr) {
        return (get_loc_point_process(_vptr));
    }
    /* Neuron setdata functions */
    extern void _nrn_setdata_reg(int, void(*)(Prop*));
    static void _setdata(Prop* _prop) {
    }
    static void _hoc_setdata(void* _vptr) {
        Prop* _prop;
        _prop = ((Point_process*)_vptr)->prop;
        _setdata(_prop);
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
    static double _hoc_use_tau2(void * _vptr);
    static double _hoc_cnst1(void * _vptr);
    static double _hoc_cnst2(void * _vptr);
    static double _hoc_tau1(void * _vptr);
    static double _hoc_tau2(void * _vptr);
    static double _hoc_table_cnst1(void * _vptr);
    static double _hoc_table_cnst2(void * _vptr);
    static double _hoc_table_tau1(void * _vptr);
    static double _hoc_table_tau2(void * _vptr);


    /* connect user functions to hoc names */
    static VoidFunc hoc_intfunc[] = {
        {0, 0}
    };
    static Member_func _member_func[] = {
        {"loc", _hoc_loc_pnt},
        {"has_loc", _hoc_has_loc},
        {"get_loc", _hoc_get_loc_pnt},
        {"use_tau2", _hoc_use_tau2},
        {"cnst1", _hoc_cnst1},
        {"cnst2", _hoc_cnst2},
        {"tau1", _hoc_tau1},
        {"tau2", _hoc_tau2},
        {"table_cnst1", _hoc_table_cnst1},
        {"table_cnst2", _hoc_table_cnst2},
        {"table_tau1", _hoc_table_tau1},
        {"table_tau2", _hoc_table_tau2},
        {nullptr, nullptr}
    };
    static double _hoc_use_tau2(void * _vptr) {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        auto* const _pnt = static_cast<Point_process*>(_vptr);
        auto* const _p = _pnt->prop;
        if (!_p) {
            hoc_execerror("POINT_PROCESS data instance not valid", nullptr);
        }
        _nrn_mechanism_cache_instance _lmc{_p};
        size_t const id{};
        _ppvar = _nrn_mechanism_access_dparam(_p);
        _thread = _extcall_thread.data();
        nt = static_cast<NrnThread*>(_pnt->_vnt);
        auto inst = make_instance_art_function_table(_p ? &_lmc : nullptr);
        double _r = 0.0;
        _r = use_tau2_art_function_table(_lmc, inst, id, _ppvar, _thread, nt, *getarg(1), *getarg(2));
        return(_r);
    }
    static double _hoc_cnst1(void * _vptr) {
        double _ret = cnst1_art_function_table(*getarg(1));
        return _ret;
    }
    static double _hoc_table_cnst1(void * _vptr) {
        double _ret = table_cnst1_art_function_table();
        return _ret;
    }
    static double _npy_cnst1(Prop* _prop) {
        return cnst1_art_function_table(*getarg(1));
    }
    static double _npy_table_cnst1(Prop* _prop) {
        return table_cnst1_art_function_table();
    }
    static double _hoc_cnst2(void * _vptr) {
        double _ret = cnst2_art_function_table(*getarg(1), *getarg(2));
        return _ret;
    }
    static double _hoc_table_cnst2(void * _vptr) {
        double _ret = table_cnst2_art_function_table();
        return _ret;
    }
    static double _npy_cnst2(Prop* _prop) {
        return cnst2_art_function_table(*getarg(1), *getarg(2));
    }
    static double _npy_table_cnst2(Prop* _prop) {
        return table_cnst2_art_function_table();
    }
    static double _hoc_tau1(void * _vptr) {
        double _ret = tau1_art_function_table(*getarg(1));
        return _ret;
    }
    static double _hoc_table_tau1(void * _vptr) {
        double _ret = table_tau1_art_function_table();
        return _ret;
    }
    static double _npy_tau1(Prop* _prop) {
        return tau1_art_function_table(*getarg(1));
    }
    static double _npy_table_tau1(Prop* _prop) {
        return table_tau1_art_function_table();
    }
    static double _hoc_tau2(void * _vptr) {
        double _ret = tau2_art_function_table(*getarg(1), *getarg(2));
        return _ret;
    }
    static double _hoc_table_tau2(void * _vptr) {
        double _ret = table_tau2_art_function_table();
        return _ret;
    }
    static double _npy_tau2(Prop* _prop) {
        return tau2_art_function_table(*getarg(1), *getarg(2));
    }
    static double _npy_table_tau2(Prop* _prop) {
        return table_tau2_art_function_table();
    }


    inline double use_tau2_art_function_table(_nrn_mechanism_cache_range& _lmc, art_function_table_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _lv, double _lx) {
        double ret_use_tau2 = 0.0;
        ret_use_tau2 = tau2_art_function_table(_lv, _lx);
        return ret_use_tau2;
    }
    double cnst1_art_function_table(double v) {
        double _arg[1];
        _arg[0] = v;
        return hoc_func_table(art_function_table_global._ptable_cnst1, 1, _arg);
    }
    double table_cnst1_art_function_table() {
        hoc_spec_table(&art_function_table_global._ptable_cnst1, 1);
        return 0.;
    }
    double cnst2_art_function_table(double v, double x) {
        double _arg[2];
        _arg[0] = v;
        _arg[1] = x;
        return hoc_func_table(art_function_table_global._ptable_cnst2, 2, _arg);
    }
    double table_cnst2_art_function_table() {
        hoc_spec_table(&art_function_table_global._ptable_cnst2, 2);
        return 0.;
    }
    double tau1_art_function_table(double v) {
        double _arg[1];
        _arg[0] = v;
        return hoc_func_table(art_function_table_global._ptable_tau1, 1, _arg);
    }
    double table_tau1_art_function_table() {
        hoc_spec_table(&art_function_table_global._ptable_tau1, 1);
        return 0.;
    }
    double tau2_art_function_table(double v, double x) {
        double _arg[2];
        _arg[0] = v;
        _arg[1] = x;
        return hoc_func_table(art_function_table_global._ptable_tau2, 2, _arg);
    }
    double table_tau2_art_function_table() {
        hoc_spec_table(&art_function_table_global._ptable_tau2, 2);
        return 0.;
    }


    static void nrn_init_art_function_table(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_art_function_table(&_lmc);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
            auto* _ppvar = _ml_arg->pdata[id];
        }
    }


    static void nrn_jacob_art_function_table(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_art_function_table(&_lmc);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
        }
    }
    static void nrn_destructor_art_function_table(Prop* prop) {
        Datum* _ppvar = _nrn_mechanism_access_dparam(prop);
        _nrn_mechanism_cache_instance _lmc{prop};
        const size_t id = 0;
        auto inst = make_instance_art_function_table(prop ? &_lmc : nullptr);

    }


    static void _initlists() {
    }


    extern "C" void _art_function_table_reg() {
        _initlists();

        _pointtype = point_register_mech(mechanism_info, nrn_alloc_art_function_table, nullptr, nullptr, nullptr, nrn_init_art_function_table, -1, 1, _hoc_create_pnt, _hoc_destroy_pnt, _member_func);

        mech_type = nrn_get_mechtype(mechanism_info[1]);
        hoc_register_parm_default(mech_type, &_parameter_defaults);
        _nrn_mechanism_register_data_fields(mech_type,
            _nrn_mechanism_field<double>{"v_unused"} /* 0 */,
            _nrn_mechanism_field<double*>{"node_area", "area"} /* 0 */,
            _nrn_mechanism_field<Point_process*>{"point_process", "pntproc"} /* 1 */
        );

        hoc_register_prop_size(mech_type, 1, 2);
        hoc_register_dparam_semantics(mech_type, 0, "area");
        hoc_register_dparam_semantics(mech_type, 1, "pntproc");
        add_nrn_artcell(mech_type, 0);
        hoc_register_var(hoc_scalar_double, hoc_vector_double, hoc_intfunc);
    }
}
