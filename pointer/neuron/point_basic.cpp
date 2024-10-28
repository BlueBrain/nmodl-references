/*********************************************************
Model Name      : point_basic
Filename        : point_basic.mod
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

static constexpr auto number_of_datum_variables = 5;
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
        "point_basic",
        0,
        "x1",
        "x2",
        "ignore",
        0,
        0,
        "p1",
        "p2",
        0
    };


    /* NEURON global variables */
    static Symbol* _ca_sym;
    static int mech_type;
    static int _pointtype;
    static _nrn_mechanism_std_vector<Datum> _extcall_thread;


    /** all global variables */
    struct point_basic_Store {
    };
    static_assert(std::is_trivially_copy_constructible_v<point_basic_Store>);
    static_assert(std::is_trivially_move_constructible_v<point_basic_Store>);
    static_assert(std::is_trivially_copy_assignable_v<point_basic_Store>);
    static_assert(std::is_trivially_move_assignable_v<point_basic_Store>);
    static_assert(std::is_trivially_destructible_v<point_basic_Store>);
    static point_basic_Store point_basic_global;
    static std::vector<double> _parameter_defaults = {
    };


    /** all mechanism instance variables and global variables */
    struct point_basic_Instance  {
        double* x1{};
        double* x2{};
        double* ignore{};
        double* ica{};
        double* v_unused{};
        const double* const* node_area{};
        const double* const* ion_ica{};
        point_basic_Store* global{&point_basic_global};
    };


    struct point_basic_NodeData  {
        int const * nodeindices;
        double const * node_voltages;
        double * node_diagonal;
        double * node_rhs;
        int nodecount;
    };


    static point_basic_Instance make_instance_point_basic(_nrn_mechanism_cache_range* _lmc) {
        if(_lmc == nullptr) {
            return point_basic_Instance();
        }

        return point_basic_Instance {
            _lmc->template fpfield_ptr<0>(),
            _lmc->template fpfield_ptr<1>(),
            _lmc->template fpfield_ptr<2>(),
            _lmc->template fpfield_ptr<3>(),
            _lmc->template fpfield_ptr<4>(),
            _lmc->template dptr_field_ptr<0>(),
            _lmc->template dptr_field_ptr<2>()
        };
    }


    static point_basic_NodeData make_node_data_point_basic(NrnThread& nt, Memb_list& _ml_arg) {
        return point_basic_NodeData {
            _ml_arg.nodeindices,
            nt.node_voltage_storage(),
            nt.node_d_storage(),
            nt.node_rhs_storage(),
            _ml_arg.nodecount
        };
    }
    static point_basic_NodeData make_node_data_point_basic(Prop * _prop) {
        if(!_prop) {
            return point_basic_NodeData();
        }

        static std::vector<int> node_index{0};
        Node* _node = _nrn_mechanism_access_node(_prop);
        return point_basic_NodeData {
            node_index.data(),
            &_nrn_mechanism_access_voltage(_node),
            &_nrn_mechanism_access_d(_node),
            &_nrn_mechanism_access_rhs(_node),
            1
        };
    }

    static void nrn_destructor_point_basic(Prop* prop);


    static void nrn_alloc_point_basic(Prop* _prop) {
        Datum *_ppvar = nullptr;
        if (nrn_point_prop_) {
            _nrn_mechanism_access_alloc_seq(_prop) = _nrn_mechanism_access_alloc_seq(nrn_point_prop_);
            _ppvar = _nrn_mechanism_access_dparam(nrn_point_prop_);
        } else {
            _ppvar = nrn_prop_datum_alloc(mech_type, 5, _prop);
            _nrn_mechanism_access_dparam(_prop) = _ppvar;
            _nrn_mechanism_cache_instance _lmc{_prop};
            size_t const _iml = 0;
            assert(_nrn_mechanism_get_num_vars(_prop) == 5);
            /*initialize range parameters*/
        }
        _nrn_mechanism_access_dparam(_prop) = _ppvar;
        Symbol * ca_sym = hoc_lookup("ca_ion");
        Prop * ca_prop = need_memb(ca_sym);
        nrn_promote(ca_prop, 0, 0);
        _ppvar[2] = _nrn_mechanism_get_param_handle(ca_prop, 3);
        if(!nrn_point_prop_) {
        }
    }


    /* Mechanism procedures and functions */
    inline static double read_p1_point_basic(_nrn_mechanism_cache_range& _lmc, point_basic_Instance& inst, point_basic_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt);
    inline static double read_p2_point_basic(_nrn_mechanism_cache_range& _lmc, point_basic_Instance& inst, point_basic_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt);
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
    static double _hoc_read_p1(void * _vptr);
    static double _hoc_read_p2(void * _vptr);


    /* connect user functions to hoc names */
    static VoidFunc hoc_intfunc[] = {
        {0, 0}
    };
    static Member_func _member_func[] = {
        {"loc", _hoc_loc_pnt},
        {"has_loc", _hoc_has_loc},
        {"get_loc", _hoc_get_loc_pnt},
        {"read_p1", _hoc_read_p1},
        {"read_p2", _hoc_read_p2},
        {nullptr, nullptr}
    };
    static double _hoc_read_p1(void * _vptr) {
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
        auto inst = make_instance_point_basic(_p ? &_lmc : nullptr);
        auto node_data = make_node_data_point_basic(_p);
        double _r = 0.0;
        _r = read_p1_point_basic(_lmc, inst, node_data, id, _ppvar, _thread, nt);
        return(_r);
    }
    static double _hoc_read_p2(void * _vptr) {
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
        auto inst = make_instance_point_basic(_p ? &_lmc : nullptr);
        auto node_data = make_node_data_point_basic(_p);
        double _r = 0.0;
        _r = read_p2_point_basic(_lmc, inst, node_data, id, _ppvar, _thread, nt);
        return(_r);
    }


    inline double read_p1_point_basic(_nrn_mechanism_cache_range& _lmc, point_basic_Instance& inst, point_basic_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt) {
        double ret_read_p1 = 0.0;
        double v = node_data.node_voltages ? node_data.node_voltages[node_data.nodeindices[id]] : 0.0;
        ret_read_p1 = (*_ppvar[3].get<double*>());
        return ret_read_p1;
    }


    inline double read_p2_point_basic(_nrn_mechanism_cache_range& _lmc, point_basic_Instance& inst, point_basic_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt) {
        double ret_read_p2 = 0.0;
        double v = node_data.node_voltages ? node_data.node_voltages[node_data.nodeindices[id]] : 0.0;
        ret_read_p2 = (*_ppvar[4].get<double*>());
        return ret_read_p2;
    }


    static void nrn_init_point_basic(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_point_basic(&_lmc);
        auto node_data = make_node_data_point_basic(*nt, *_ml_arg);
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


    static void nrn_jacob_point_basic(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_point_basic(&_lmc);
        auto node_data = make_node_data_point_basic(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
        }
    }
    static void nrn_destructor_point_basic(Prop* prop) {
        Datum* _ppvar = _nrn_mechanism_access_dparam(prop);
        _nrn_mechanism_cache_instance _lmc{prop};
        const size_t id = 0;
        auto inst = make_instance_point_basic(prop ? &_lmc : nullptr);
        auto node_data = make_node_data_point_basic(prop);

    }


    static void _initlists() {
    }


    extern "C" void _point_basic_reg() {
        _initlists();

        ion_reg("ca", -10000);

        _ca_sym = hoc_lookup("ca_ion");

        _pointtype = point_register_mech(mechanism_info, nrn_alloc_point_basic, nullptr, nullptr, nullptr, nrn_init_point_basic, 3, 1, _hoc_create_pnt, _hoc_destroy_pnt, _member_func);

        mech_type = nrn_get_mechtype(mechanism_info[1]);
        hoc_register_parm_default(mech_type, &_parameter_defaults);
        _nrn_mechanism_register_data_fields(mech_type,
            _nrn_mechanism_field<double>{"x1"} /* 0 */,
            _nrn_mechanism_field<double>{"x2"} /* 1 */,
            _nrn_mechanism_field<double>{"ignore"} /* 2 */,
            _nrn_mechanism_field<double>{"ica"} /* 3 */,
            _nrn_mechanism_field<double>{"v_unused"} /* 4 */,
            _nrn_mechanism_field<double*>{"node_area", "area"} /* 0 */,
            _nrn_mechanism_field<Point_process*>{"point_process", "pntproc"} /* 1 */,
            _nrn_mechanism_field<double*>{"ion_ica", "ca_ion"} /* 2 */,
            _nrn_mechanism_field<double*>{"p1", "pointer"} /* 3 */,
            _nrn_mechanism_field<double*>{"p2", "pointer"} /* 4 */
        );

        hoc_register_prop_size(mech_type, 5, 5);
        hoc_register_dparam_semantics(mech_type, 0, "area");
        hoc_register_dparam_semantics(mech_type, 1, "pntproc");
        hoc_register_dparam_semantics(mech_type, 2, "ca_ion");
        hoc_register_dparam_semantics(mech_type, 3, "pointer");
        hoc_register_dparam_semantics(mech_type, 4, "pointer");
        hoc_register_var(hoc_scalar_double, hoc_vector_double, hoc_intfunc);
    }
}
