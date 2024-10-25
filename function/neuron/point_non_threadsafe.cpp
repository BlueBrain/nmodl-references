/*********************************************************
Model Name      : point_non_threadsafe
Filename        : point_non_threadsafe.mod
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

static constexpr auto number_of_datum_variables = 2;
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

extern Prop* nrn_point_prop_;
extern Node* nrn_alloc_node_;


namespace neuron {
    #ifndef NRN_PRCELLSTATE
    #define NRN_PRCELLSTATE 0
    #endif


    /** channel information */
    static const char *mechanism_info[] = {
        "7.7.0",
        "point_non_threadsafe",
        0,
        "x",
        0,
        "z",
        0,
        0
    };


    /* NEURON global variables */
    static int mech_type;
    static int _pointtype;
    static _nrn_mechanism_std_vector<Datum> _extcall_thread;


    /** all global variables */
    struct point_non_threadsafe_Store {
        double gbl{0};
        double z0{0};
    };
    static_assert(std::is_trivially_copy_constructible_v<point_non_threadsafe_Store>);
    static_assert(std::is_trivially_move_constructible_v<point_non_threadsafe_Store>);
    static_assert(std::is_trivially_copy_assignable_v<point_non_threadsafe_Store>);
    static_assert(std::is_trivially_move_assignable_v<point_non_threadsafe_Store>);
    static_assert(std::is_trivially_destructible_v<point_non_threadsafe_Store>);
    static point_non_threadsafe_Store point_non_threadsafe_global;
    auto gbl_point_non_threadsafe() -> std::decay<decltype(point_non_threadsafe_global.gbl)>::type  {
        return point_non_threadsafe_global.gbl;
    }
    auto z0_point_non_threadsafe() -> std::decay<decltype(point_non_threadsafe_global.z0)>::type  {
        return point_non_threadsafe_global.z0;
    }

    static std::vector<double> _parameter_defaults = {
    };


    /** all mechanism instance variables and global variables */
    struct point_non_threadsafe_Instance  {
        double* x{};
        double* z{};
        double* Dz{};
        double* v_unused{};
        const double* const* node_area{};
        point_non_threadsafe_Store* global{&point_non_threadsafe_global};
    };


    struct point_non_threadsafe_NodeData  {
        int const * nodeindices;
        double const * node_voltages;
        double * node_diagonal;
        double * node_rhs;
        int nodecount;
    };


    static point_non_threadsafe_Instance make_instance_point_non_threadsafe(_nrn_mechanism_cache_range* _lmc) {
        if(_lmc == nullptr) {
            return point_non_threadsafe_Instance();
        }

        return point_non_threadsafe_Instance {
            _lmc->template fpfield_ptr<0>(),
            _lmc->template fpfield_ptr<1>(),
            _lmc->template fpfield_ptr<2>(),
            _lmc->template fpfield_ptr<3>(),
            _lmc->template dptr_field_ptr<0>()
        };
    }


    static point_non_threadsafe_NodeData make_node_data_point_non_threadsafe(NrnThread& nt, Memb_list& _ml_arg) {
        return point_non_threadsafe_NodeData {
            _ml_arg.nodeindices,
            nt.node_voltage_storage(),
            nt.node_d_storage(),
            nt.node_rhs_storage(),
            _ml_arg.nodecount
        };
    }
    static point_non_threadsafe_NodeData make_node_data_point_non_threadsafe(Prop * _prop) {
        if(!_prop) {
            return point_non_threadsafe_NodeData();
        }

        static std::vector<int> node_index{0};
        Node* _node = _nrn_mechanism_access_node(_prop);
        return point_non_threadsafe_NodeData {
            node_index.data(),
            &_nrn_mechanism_access_voltage(_node),
            &_nrn_mechanism_access_d(_node),
            &_nrn_mechanism_access_rhs(_node),
            1
        };
    }

    static void nrn_destructor_point_non_threadsafe(Prop* prop);


    static void nrn_alloc_point_non_threadsafe(Prop* _prop) {
        Datum *_ppvar = nullptr;
        if (nrn_point_prop_) {
            _nrn_mechanism_access_alloc_seq(_prop) = _nrn_mechanism_access_alloc_seq(nrn_point_prop_);
            _ppvar = _nrn_mechanism_access_dparam(nrn_point_prop_);
        } else {
            _ppvar = nrn_prop_datum_alloc(mech_type, 2, _prop);
            _nrn_mechanism_access_dparam(_prop) = _ppvar;
            _nrn_mechanism_cache_instance _lmc{_prop};
            size_t const _iml = 0;
            assert(_nrn_mechanism_get_num_vars(_prop) == 4);
            /*initialize range parameters*/
        }
        _nrn_mechanism_access_dparam(_prop) = _ppvar;
        if(!nrn_point_prop_) {
        }
    }


    /* Mechanism procedures and functions */
    inline static double x_plus_a_point_non_threadsafe(_nrn_mechanism_cache_range& _lmc, point_non_threadsafe_Instance& inst, point_non_threadsafe_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _la);
    inline static double v_plus_a_point_non_threadsafe(_nrn_mechanism_cache_range& _lmc, point_non_threadsafe_Instance& inst, point_non_threadsafe_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _la);
    inline static double identity_point_non_threadsafe(_nrn_mechanism_cache_range& _lmc, point_non_threadsafe_Instance& inst, point_non_threadsafe_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _lv);
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
        {"gbl_point_non_threadsafe", &point_non_threadsafe_global.gbl},
        {nullptr, nullptr}
    };


    /** connect global (array) variables to hoc -- */
    static DoubVec hoc_vector_double[] = {
        {nullptr, nullptr, 0}
    };


    /* declaration of user functions */
    static double _hoc_x_plus_a(void * _vptr);
    static double _hoc_v_plus_a(void * _vptr);
    static double _hoc_identity(void * _vptr);


    /* connect user functions to hoc names */
    static VoidFunc hoc_intfunc[] = {
        {0, 0}
    };
    static Member_func _member_func[] = {
        {"loc", _hoc_loc_pnt},
        {"has_loc", _hoc_has_loc},
        {"get_loc", _hoc_get_loc_pnt},
        {"x_plus_a", _hoc_x_plus_a},
        {"v_plus_a", _hoc_v_plus_a},
        {"identity", _hoc_identity},
        {nullptr, nullptr}
    };
    static double _hoc_x_plus_a(void * _vptr) {
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
        auto inst = make_instance_point_non_threadsafe(_p ? &_lmc : nullptr);
        auto node_data = make_node_data_point_non_threadsafe(_p);
        double _r = 0.0;
        _r = x_plus_a_point_non_threadsafe(_lmc, inst, node_data, id, _ppvar, _thread, nt, *getarg(1));
        return(_r);
    }
    static double _hoc_v_plus_a(void * _vptr) {
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
        auto inst = make_instance_point_non_threadsafe(_p ? &_lmc : nullptr);
        auto node_data = make_node_data_point_non_threadsafe(_p);
        double _r = 0.0;
        _r = v_plus_a_point_non_threadsafe(_lmc, inst, node_data, id, _ppvar, _thread, nt, *getarg(1));
        return(_r);
    }
    static double _hoc_identity(void * _vptr) {
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
        auto inst = make_instance_point_non_threadsafe(_p ? &_lmc : nullptr);
        auto node_data = make_node_data_point_non_threadsafe(_p);
        double _r = 0.0;
        _r = identity_point_non_threadsafe(_lmc, inst, node_data, id, _ppvar, _thread, nt, *getarg(1));
        return(_r);
    }


    inline double x_plus_a_point_non_threadsafe(_nrn_mechanism_cache_range& _lmc, point_non_threadsafe_Instance& inst, point_non_threadsafe_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _la) {
        double ret_x_plus_a = 0.0;
        double v = node_data.node_voltages ? node_data.node_voltages[node_data.nodeindices[id]] : 0.0;
        ret_x_plus_a = inst.x[id] + _la;
        return ret_x_plus_a;
    }


    inline double v_plus_a_point_non_threadsafe(_nrn_mechanism_cache_range& _lmc, point_non_threadsafe_Instance& inst, point_non_threadsafe_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _la) {
        double ret_v_plus_a = 0.0;
        double v = node_data.node_voltages ? node_data.node_voltages[node_data.nodeindices[id]] : 0.0;
        ret_v_plus_a = inst.v_unused[id] + _la;
        return ret_v_plus_a;
    }


    inline double identity_point_non_threadsafe(_nrn_mechanism_cache_range& _lmc, point_non_threadsafe_Instance& inst, point_non_threadsafe_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _lv) {
        double ret_identity = 0.0;
        double v = node_data.node_voltages ? node_data.node_voltages[node_data.nodeindices[id]] : 0.0;
        ret_identity = _lv;
        return ret_identity;
    }


    static void nrn_init_point_non_threadsafe(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_point_non_threadsafe(&_lmc);
        auto node_data = make_node_data_point_non_threadsafe(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
            auto* _ppvar = _ml_arg->pdata[id];
            int node_id = node_data.nodeindices[id];
            inst.v_unused[id] = node_data.node_voltages[node_id];
            inst.z[id] = inst.global->z0;
            inst.x[id] = 1.0;
            inst.global->gbl = 42.0;
        }
    }


    static void nrn_jacob_point_non_threadsafe(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_point_non_threadsafe(&_lmc);
        auto node_data = make_node_data_point_non_threadsafe(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
        }
    }
    static void nrn_destructor_point_non_threadsafe(Prop* prop) {
        Datum* _ppvar = _nrn_mechanism_access_dparam(prop);
        _nrn_mechanism_cache_instance _lmc{prop};
        const size_t id = 0;
        auto inst = make_instance_point_non_threadsafe(prop ? &_lmc : nullptr);
        auto node_data = make_node_data_point_non_threadsafe(prop);

    }


    static void _initlists() {
    }


    extern "C" void _point_non_threadsafe_reg() {
        _initlists();

        _pointtype = point_register_mech(mechanism_info, nrn_alloc_point_non_threadsafe, nullptr, nullptr, nullptr, nrn_init_point_non_threadsafe, -1, 1, _hoc_create_pnt, _hoc_destroy_pnt, _member_func);

        mech_type = nrn_get_mechtype(mechanism_info[1]);
        hoc_register_parm_default(mech_type, &_parameter_defaults);
        _nrn_mechanism_register_data_fields(mech_type,
            _nrn_mechanism_field<double>{"x"} /* 0 */,
            _nrn_mechanism_field<double>{"z"} /* 1 */,
            _nrn_mechanism_field<double>{"Dz"} /* 2 */,
            _nrn_mechanism_field<double>{"v_unused"} /* 3 */,
            _nrn_mechanism_field<double*>{"node_area", "area"} /* 0 */,
            _nrn_mechanism_field<Point_process*>{"point_process", "pntproc"} /* 1 */
        );

        hoc_register_prop_size(mech_type, 4, 2);
        hoc_register_dparam_semantics(mech_type, 0, "area");
        hoc_register_dparam_semantics(mech_type, 1, "pntproc");
        hoc_register_var(hoc_scalar_double, hoc_vector_double, hoc_intfunc);
    }
}
