/*********************************************************
Model Name      : test_func_proc_pnt
Filename        : func_proc_pnt.mod
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
extern void _nrn_thread_reg(int, int, void(*)(Datum*));

/* NEURON global macro definitions */
/* VECTORIZED */
#define NRN_VECTORIZED 1

static constexpr auto number_of_datum_variables = 2;
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

extern Prop* nrn_point_prop_;


namespace neuron {
    #ifndef NRN_PRCELLSTATE
    #define NRN_PRCELLSTATE 0
    #endif


    /** channel information */
    static const char *mechanism_info[] = {
        "7.7.0",
        "test_func_proc_pnt",
        0,
        "x",
        0,
        0,
        0
    };


    /* NEURON global variables */
    static int mech_type;
    static int _pointtype;
    static int hoc_nrnpointerindex = -1;
    static _nrn_mechanism_std_vector<Datum> _extcall_thread;


    /** all global variables */
    struct test_func_proc_pnt_Store {
    };
    static_assert(std::is_trivially_copy_constructible_v<test_func_proc_pnt_Store>);
    static_assert(std::is_trivially_move_constructible_v<test_func_proc_pnt_Store>);
    static_assert(std::is_trivially_copy_assignable_v<test_func_proc_pnt_Store>);
    static_assert(std::is_trivially_move_assignable_v<test_func_proc_pnt_Store>);
    static_assert(std::is_trivially_destructible_v<test_func_proc_pnt_Store>);
    test_func_proc_pnt_Store test_func_proc_pnt_global;


    /** all mechanism instance variables and global variables */
    struct test_func_proc_pnt_Instance  {
        double* x{};
        double* v_unused{};
        const double* const* node_area{};
        test_func_proc_pnt_Store* global{&test_func_proc_pnt_global};
    };


    struct test_func_proc_pnt_NodeData  {
        int const * nodeindices;
        double const * node_voltages;
        double * node_diagonal;
        double * node_rhs;
        int nodecount;
    };


    static test_func_proc_pnt_Instance make_instance_test_func_proc_pnt(_nrn_mechanism_cache_range& _ml) {
        return test_func_proc_pnt_Instance {
            _ml.template fpfield_ptr<0>(),
            _ml.template fpfield_ptr<1>(),
            _ml.template dptr_field_ptr<0>()
        };
    }


    static test_func_proc_pnt_NodeData make_node_data_test_func_proc_pnt(NrnThread& _nt, Memb_list& _ml_arg) {
        return test_func_proc_pnt_NodeData {
            _ml_arg.nodeindices,
            _nt.node_voltage_storage(),
            _nt.node_d_storage(),
            _nt.node_rhs_storage(),
            _ml_arg.nodecount
        };
    }


    static void nrn_alloc_test_func_proc_pnt(Prop* _prop) {
        Prop *prop_ion{};
        Datum *_ppvar{};
        if (nrn_point_prop_) {
            _nrn_mechanism_access_alloc_seq(_prop) = _nrn_mechanism_access_alloc_seq(nrn_point_prop_);
            _ppvar = _nrn_mechanism_access_dparam(nrn_point_prop_);
        } else {
            _ppvar = nrn_prop_datum_alloc(mech_type, 2, _prop);
            _nrn_mechanism_access_dparam(_prop) = _ppvar;
            _nrn_mechanism_cache_instance _ml_real{_prop};
            auto* const _ml = &_ml_real;
            size_t const _iml{};
            assert(_nrn_mechanism_get_num_vars(_prop) == 2);
            /*initialize range parameters*/
        }
        _nrn_mechanism_access_dparam(_prop) = _ppvar;
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
    /* Mechanism procedures and functions */
    inline double x_plus_a_test_func_proc_pnt(_nrn_mechanism_cache_range* _ml, test_func_proc_pnt_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double a);
    inline int set_x_42_test_func_proc_pnt(_nrn_mechanism_cache_range* _ml, test_func_proc_pnt_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* _nt);
    inline int set_x_a_test_func_proc_pnt(_nrn_mechanism_cache_range* _ml, test_func_proc_pnt_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double a);


    /** connect global (scalar) variables to hoc -- */
    static DoubScal hoc_scalar_double[] = {
        {nullptr, nullptr}
    };


    /** connect global (array) variables to hoc -- */
    static DoubVec hoc_vector_double[] = {
        {nullptr, nullptr, 0}
    };


    /* declaration of user functions */
    static double _hoc_set_x_42(void*);
    static double _hoc_set_x_a(void*);
    static double _hoc_x_plus_a(void*);


    /* connect user functions to hoc names */
    static VoidFunc hoc_intfunc[] = {
        {0, 0}
    };
    static Member_func _member_func[] = {
        {"loc", _hoc_loc_pnt},
        {"has_loc", _hoc_has_loc},
        {"get_loc", _hoc_get_loc_pnt},
        {"set_x_42", _hoc_set_x_42},
        {"set_x_a", _hoc_set_x_a},
        {"x_plus_a", _hoc_x_plus_a},
        {0, 0}
    };
    static double _hoc_set_x_42(void* _vptr) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* _nt;
        auto* const _pnt = static_cast<Point_process*>(_vptr);
        auto* const _p = _pnt->prop;
        if (!_p) {
            hoc_execerror("POINT_PROCESS data instance not valid", NULL);
        }
        _nrn_mechanism_cache_instance _ml_real{_p};
        auto* const _ml = &_ml_real;
        size_t const id{};
        _ppvar = _nrn_mechanism_access_dparam(_p);
        _thread = _extcall_thread.data();
        _nt = static_cast<NrnThread*>(_pnt->_vnt);
        auto inst = make_instance_test_func_proc_pnt(_ml_real);
        _r = 1.;
        set_x_42_test_func_proc_pnt(_ml, inst, id, _ppvar, _thread, _nt);
        return(_r);
    }
    static double _hoc_set_x_a(void* _vptr) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* _nt;
        auto* const _pnt = static_cast<Point_process*>(_vptr);
        auto* const _p = _pnt->prop;
        if (!_p) {
            hoc_execerror("POINT_PROCESS data instance not valid", NULL);
        }
        _nrn_mechanism_cache_instance _ml_real{_p};
        auto* const _ml = &_ml_real;
        size_t const id{};
        _ppvar = _nrn_mechanism_access_dparam(_p);
        _thread = _extcall_thread.data();
        _nt = static_cast<NrnThread*>(_pnt->_vnt);
        auto inst = make_instance_test_func_proc_pnt(_ml_real);
        _r = 1.;
        set_x_a_test_func_proc_pnt(_ml, inst, id, _ppvar, _thread, _nt, *getarg(1));
        return(_r);
    }
    static double _hoc_x_plus_a(void* _vptr) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* _nt;
        auto* const _pnt = static_cast<Point_process*>(_vptr);
        auto* const _p = _pnt->prop;
        if (!_p) {
            hoc_execerror("POINT_PROCESS data instance not valid", NULL);
        }
        _nrn_mechanism_cache_instance _ml_real{_p};
        auto* const _ml = &_ml_real;
        size_t const id{};
        _ppvar = _nrn_mechanism_access_dparam(_p);
        _thread = _extcall_thread.data();
        _nt = static_cast<NrnThread*>(_pnt->_vnt);
        auto inst = make_instance_test_func_proc_pnt(_ml_real);
        _r = x_plus_a_test_func_proc_pnt(_ml, inst, id, _ppvar, _thread, _nt, *getarg(1));
        return(_r);
    }


    inline int set_x_42_test_func_proc_pnt(_nrn_mechanism_cache_range* _ml, test_func_proc_pnt_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
        int ret_set_x_42 = 0;
        auto v = inst.v_unused[id];
        inst.x[id] = 42.0;
        return ret_set_x_42;
    }


    inline int set_x_a_test_func_proc_pnt(_nrn_mechanism_cache_range* _ml, test_func_proc_pnt_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double a) {
        int ret_set_x_a = 0;
        auto v = inst.v_unused[id];
        inst.x[id] = a;
        return ret_set_x_a;
    }


    inline double x_plus_a_test_func_proc_pnt(_nrn_mechanism_cache_range* _ml, test_func_proc_pnt_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double a) {
        double ret_x_plus_a = 0.0;
        auto v = inst.v_unused[id];
        ret_x_plus_a = inst.x[id] + a;
        return ret_x_plus_a;
    }


    void nrn_init_test_func_proc_pnt(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
        auto inst = make_instance_test_func_proc_pnt(_lmr);
        auto node_data = make_node_data_test_func_proc_pnt(*_nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* const _ml = &_lmr;
        auto* _thread = _ml_arg->_thread;
        double * _thread_globals = nullptr;
        for (int id = 0; id < nodecount; id++) {
            auto* _ppvar = _ml_arg->pdata[id];
            int node_id = node_data.nodeindices[id];
            auto v = node_data.node_voltages[node_id];
            inst.v_unused[id] = v;
        }
    }


    static void nrn_jacob_test_func_proc_pnt(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
        auto inst = make_instance_test_func_proc_pnt(_lmr);
        auto node_data = make_node_data_test_func_proc_pnt(*_nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
        }
    }


    static void _initlists() {
    }


    /** register channel with the simulator */
    extern "C" void _func_proc_pnt_reg() {
        _initlists();



        _pointtype = point_register_mech(mechanism_info, nrn_alloc_test_func_proc_pnt, nullptr, nullptr, nullptr, nrn_init_test_func_proc_pnt, hoc_nrnpointerindex, 1, _hoc_create_pnt, _hoc_destroy_pnt, _member_func);

        mech_type = nrn_get_mechtype(mechanism_info[1]);
        _nrn_mechanism_register_data_fields(mech_type,
            _nrn_mechanism_field<double>{"x"} /* 0 */,
            _nrn_mechanism_field<double>{"v_unused"} /* 1 */,
            _nrn_mechanism_field<double*>{"node_area", "area"} /* 0 */,
            _nrn_mechanism_field<Point_process*>{"point_process", "pntproc"} /* 1 */
        );

        hoc_register_prop_size(mech_type, 2, 2);
        hoc_register_dparam_semantics(mech_type, 0, "area");
        hoc_register_dparam_semantics(mech_type, 1, "pntproc");
        hoc_register_var(hoc_scalar_double, hoc_vector_double, hoc_intfunc);
    }
}
