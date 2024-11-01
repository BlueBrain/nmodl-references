/*********************************************************
Model Name      : SnapSyn
Filename        : snapsyn.mod
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
        "SnapSyn",
        "e",
        0,
        "i",
        0,
        0,
        0
    };


    /* NEURON global variables */
    static int mech_type;
    static int _pointtype;
    static _nrn_mechanism_std_vector<Datum> _extcall_thread;


    /** all global variables */
    struct SnapSyn_Store {
    };
    static_assert(std::is_trivially_copy_constructible_v<SnapSyn_Store>);
    static_assert(std::is_trivially_move_constructible_v<SnapSyn_Store>);
    static_assert(std::is_trivially_copy_assignable_v<SnapSyn_Store>);
    static_assert(std::is_trivially_move_assignable_v<SnapSyn_Store>);
    static_assert(std::is_trivially_destructible_v<SnapSyn_Store>);
    static SnapSyn_Store SnapSyn_global;
    static std::vector<double> _parameter_defaults = {
        10 /* e */
    };


    /** all mechanism instance variables and global variables */
    struct SnapSyn_Instance  {
        double* e{};
        double* i{};
        double* g{};
        double* v_unused{};
        double* g_unused{};
        double* tsave{};
        const double* const* node_area{};
        SnapSyn_Store* global{&SnapSyn_global};
    };


    struct SnapSyn_NodeData  {
        int const * nodeindices;
        double const * node_voltages;
        double * node_diagonal;
        double * node_rhs;
        int nodecount;
    };


    static SnapSyn_Instance make_instance_SnapSyn(_nrn_mechanism_cache_range* _lmc) {
        if(_lmc == nullptr) {
            return SnapSyn_Instance();
        }

        return SnapSyn_Instance {
            _lmc->template fpfield_ptr<0>(),
            _lmc->template fpfield_ptr<1>(),
            _lmc->template fpfield_ptr<2>(),
            _lmc->template fpfield_ptr<3>(),
            _lmc->template fpfield_ptr<4>(),
            _lmc->template fpfield_ptr<5>(),
            _lmc->template dptr_field_ptr<0>()
        };
    }


    static SnapSyn_NodeData make_node_data_SnapSyn(NrnThread& nt, Memb_list& _ml_arg) {
        return SnapSyn_NodeData {
            _ml_arg.nodeindices,
            nt.node_voltage_storage(),
            nt.node_d_storage(),
            nt.node_rhs_storage(),
            _ml_arg.nodecount
        };
    }
    static SnapSyn_NodeData make_node_data_SnapSyn(Prop * _prop) {
        if(!_prop) {
            return SnapSyn_NodeData();
        }

        static std::vector<int> node_index{0};
        Node* _node = _nrn_mechanism_access_node(_prop);
        return SnapSyn_NodeData {
            node_index.data(),
            &_nrn_mechanism_access_voltage(_node),
            &_nrn_mechanism_access_d(_node),
            &_nrn_mechanism_access_rhs(_node),
            1
        };
    }

    static void nrn_destructor_SnapSyn(Prop* prop);


    static void nrn_alloc_SnapSyn(Prop* _prop) {
        Datum *_ppvar = nullptr;
        if (nrn_point_prop_) {
            _nrn_mechanism_access_alloc_seq(_prop) = _nrn_mechanism_access_alloc_seq(nrn_point_prop_);
            _ppvar = _nrn_mechanism_access_dparam(nrn_point_prop_);
        } else {
            _ppvar = nrn_prop_datum_alloc(mech_type, 2, _prop);
            _nrn_mechanism_access_dparam(_prop) = _ppvar;
            _nrn_mechanism_cache_instance _lmc{_prop};
            size_t const _iml = 0;
            assert(_nrn_mechanism_get_num_vars(_prop) == 6);
            /*initialize range parameters*/
            _lmc.template fpfield<0>(_iml) = _parameter_defaults[0]; /* e */
        }
        _nrn_mechanism_access_dparam(_prop) = _ppvar;
        if(!nrn_point_prop_) {
        }
    }


    /* Mechanism procedures and functions */
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


    /* connect user functions to hoc names */
    static VoidFunc hoc_intfunc[] = {
        {0, 0}
    };
    static Member_func _member_func[] = {
        {"loc", _hoc_loc_pnt},
        {"has_loc", _hoc_has_loc},
        {"get_loc", _hoc_get_loc_pnt},
        {nullptr, nullptr}
    };


    static void nrn_init_SnapSyn(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_SnapSyn(&_lmc);
        auto node_data = make_node_data_SnapSyn(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        #pragma omp simd
        #pragma ivdep
        for (int id = 0; id < nodecount; id++) {
            auto* _ppvar = _ml_arg->pdata[id];
            int node_id = node_data.nodeindices[id];
            inst.v_unused[id] = node_data.node_voltages[node_id];
            inst.g[id] = 0.0;
        }
    }


    static inline double nrn_current_SnapSyn(_nrn_mechanism_cache_range& _lmc, NrnThread* nt, Datum* _ppvar, Datum* _thread, size_t id, SnapSyn_Instance& inst, SnapSyn_NodeData& node_data, double v) {
        inst.v_unused[id] = v;
        double current = 0.0;
        inst.i[id] = inst.g[id] * (inst.v_unused[id] - inst.e[id]);
        current += inst.i[id];
        return current;
    }


    /** update current */
    static void nrn_cur_SnapSyn(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_SnapSyn(&_lmc);
        auto node_data = make_node_data_SnapSyn(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        #pragma omp simd
        #pragma ivdep
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_data.nodeindices[id];
            double v = node_data.node_voltages[node_id];
            auto* _ppvar = _ml_arg->pdata[id];
            double I1 = nrn_current_SnapSyn(_lmc, nt, _ppvar, _thread, id, inst, node_data, v+0.001);
            double I0 = nrn_current_SnapSyn(_lmc, nt, _ppvar, _thread, id, inst, node_data, v);
            double rhs = I0;
            double g = (I1-I0)/0.001;
            double mfactor = 1.e2/(*inst.node_area[id]);
            g = g*mfactor;
            rhs = rhs*mfactor;
            node_data.node_rhs[node_id] -= rhs;
            inst.g_unused[id] = g;
        }
    }


    static void nrn_state_SnapSyn(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_SnapSyn(&_lmc);
        auto node_data = make_node_data_SnapSyn(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        #pragma omp simd
        #pragma ivdep
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_data.nodeindices[id];
            auto* _ppvar = _ml_arg->pdata[id];
            inst.v_unused[id] = node_data.node_voltages[node_id];
        }
    }


    static void nrn_jacob_SnapSyn(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_SnapSyn(&_lmc);
        auto node_data = make_node_data_SnapSyn(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        #pragma omp simd
        #pragma ivdep
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_data.nodeindices[id];
            node_data.node_diagonal[node_id] += inst.g_unused[id];
        }
    }
    static void nrn_net_receive_SnapSyn(Point_process* _pnt, double* _args, double flag) {
        _nrn_mechanism_cache_instance _lmc{_pnt->prop};
        auto * nt = static_cast<NrnThread*>(_pnt->_vnt);
        auto * _ppvar = _nrn_mechanism_access_dparam(_pnt->prop);
        auto inst = make_instance_SnapSyn(&_lmc);
        auto node_data = make_node_data_SnapSyn(_pnt->prop);
        // nocmodl has a nullptr dereference for thread variables.
        // NMODL will fail to compile at a later point, because of
        // missing '_thread_vars'.
        Datum * _thread = nullptr;
        size_t id = 0;
        double t = nt->_t;
        inst.g[id] = inst.g[id] + _args[0];

    }
    static void nrn_destructor_SnapSyn(Prop* prop) {
        Datum* _ppvar = _nrn_mechanism_access_dparam(prop);
        _nrn_mechanism_cache_instance _lmc{prop};
        const size_t id = 0;
        auto inst = make_instance_SnapSyn(prop ? &_lmc : nullptr);
        auto node_data = make_node_data_SnapSyn(prop);

    }


    static void _initlists() {
    }


    extern "C" void _snapsyn_reg() {
        _initlists();

        _pointtype = point_register_mech(mechanism_info, nrn_alloc_SnapSyn, nrn_cur_SnapSyn, nrn_jacob_SnapSyn, nrn_state_SnapSyn, nrn_init_SnapSyn, -1, 1, _hoc_create_pnt, _hoc_destroy_pnt, _member_func);

        mech_type = nrn_get_mechtype(mechanism_info[1]);
        hoc_register_parm_default(mech_type, &_parameter_defaults);
        _nrn_mechanism_register_data_fields(mech_type,
            _nrn_mechanism_field<double>{"e"} /* 0 */,
            _nrn_mechanism_field<double>{"i"} /* 1 */,
            _nrn_mechanism_field<double>{"g"} /* 2 */,
            _nrn_mechanism_field<double>{"v_unused"} /* 3 */,
            _nrn_mechanism_field<double>{"g_unused"} /* 4 */,
            _nrn_mechanism_field<double>{"tsave"} /* 5 */,
            _nrn_mechanism_field<double*>{"node_area", "area"} /* 0 */,
            _nrn_mechanism_field<Point_process*>{"point_process", "pntproc"} /* 1 */
        );

        hoc_register_prop_size(mech_type, 6, 2);
        hoc_register_dparam_semantics(mech_type, 0, "area");
        hoc_register_dparam_semantics(mech_type, 1, "pntproc");
        hoc_register_var(hoc_scalar_double, hoc_vector_double, hoc_intfunc);
        pnt_receive[mech_type] = nrn_net_receive_SnapSyn;
        pnt_receive_size[mech_type] = 1;
    }
}
