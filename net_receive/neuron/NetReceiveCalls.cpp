/*********************************************************
Model Name      : NetReceiveCalls
Filename        : NetReceiveCalls.mod
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
extern void _cvode_abstol(Symbol**, double*, int);
extern Node* nrn_alloc_node_;


namespace neuron {
    #ifndef NRN_PRCELLSTATE
    #define NRN_PRCELLSTATE 0
    #endif


    /** channel information */
    static const char *mechanism_info[] = {
        "7.7.0",
        "NetReceiveCalls",
        0,
        "c1",
        "c2",
        0,
        0,
        0
    };


    /* NEURON global variables */
    static int mech_type;
    static int _pointtype;
    static _nrn_mechanism_std_vector<Datum> _extcall_thread;


    /** all global variables */
    struct NetReceiveCalls_Store {
    };
    static_assert(std::is_trivially_copy_constructible_v<NetReceiveCalls_Store>);
    static_assert(std::is_trivially_move_constructible_v<NetReceiveCalls_Store>);
    static_assert(std::is_trivially_copy_assignable_v<NetReceiveCalls_Store>);
    static_assert(std::is_trivially_move_assignable_v<NetReceiveCalls_Store>);
    static_assert(std::is_trivially_destructible_v<NetReceiveCalls_Store>);
    static NetReceiveCalls_Store NetReceiveCalls_global;
    static std::vector<double> _parameter_defaults = {
    };


    /** all mechanism instance variables and global variables */
    struct NetReceiveCalls_Instance  {
        double* c1{};
        double* c2{};
        double* v_unused{};
        double* tsave{};
        const double* const* node_area{};
        NetReceiveCalls_Store* global{&NetReceiveCalls_global};
    };


    struct NetReceiveCalls_NodeData  {
        int const * nodeindices;
        double const * node_voltages;
        double * node_diagonal;
        double * node_rhs;
        int nodecount;
    };


    static NetReceiveCalls_Instance make_instance_NetReceiveCalls(_nrn_mechanism_cache_range& _lmc) {
        return NetReceiveCalls_Instance {
            _lmc.template fpfield_ptr<0>(),
            _lmc.template fpfield_ptr<1>(),
            _lmc.template fpfield_ptr<2>(),
            _lmc.template fpfield_ptr<3>(),
            _lmc.template dptr_field_ptr<0>()
        };
    }


    static NetReceiveCalls_NodeData make_node_data_NetReceiveCalls(NrnThread& nt, Memb_list& _ml_arg) {
        return NetReceiveCalls_NodeData {
            _ml_arg.nodeindices,
            nt.node_voltage_storage(),
            nt.node_d_storage(),
            nt.node_rhs_storage(),
            _ml_arg.nodecount
        };
    }
    static NetReceiveCalls_NodeData make_node_data_NetReceiveCalls(Prop * _prop) {
        static std::vector<int> node_index{0};
        Node* _node = _nrn_mechanism_access_node(_prop);
        return NetReceiveCalls_NodeData {
            node_index.data(),
            &_nrn_mechanism_access_voltage(_node),
            &_nrn_mechanism_access_d(_node),
            &_nrn_mechanism_access_rhs(_node),
            1
        };
    }

    static void nrn_destructor_NetReceiveCalls(Prop* prop);


    static void nrn_alloc_NetReceiveCalls(Prop* _prop) {
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
    inline static double one_NetReceiveCalls(_nrn_mechanism_cache_range& _lmc, NetReceiveCalls_Instance& inst, NetReceiveCalls_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt);
    inline static int increment_c2_NetReceiveCalls(_nrn_mechanism_cache_range& _lmc, NetReceiveCalls_Instance& inst, NetReceiveCalls_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt);
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
    static double _hoc_one(void * _vptr);
    static double _hoc_increment_c2(void * _vptr);


    /* connect user functions to hoc names */
    static VoidFunc hoc_intfunc[] = {
        {0, 0}
    };
    static Member_func _member_func[] = {
        {"loc", _hoc_loc_pnt},
        {"has_loc", _hoc_has_loc},
        {"get_loc", _hoc_get_loc_pnt},
        {"increment_c2", _hoc_increment_c2},
        {"one", _hoc_one},
        {nullptr, nullptr}
    };
    static double _hoc_increment_c2(void * _vptr) {
        double _r{};
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
        auto inst = make_instance_NetReceiveCalls(_lmc);
        auto node_data = make_node_data_NetReceiveCalls(_p);
        _r = 1.;
        increment_c2_NetReceiveCalls(_lmc, inst, node_data, id, _ppvar, _thread, nt);
        return(_r);
    }
    static double _hoc_one(void * _vptr) {
        double _r{};
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
        auto inst = make_instance_NetReceiveCalls(_lmc);
        auto node_data = make_node_data_NetReceiveCalls(_p);
        _r = one_NetReceiveCalls(_lmc, inst, node_data, id, _ppvar, _thread, nt);
        return(_r);
    }


    inline int increment_c2_NetReceiveCalls(_nrn_mechanism_cache_range& _lmc, NetReceiveCalls_Instance& inst, NetReceiveCalls_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt) {
        int ret_increment_c2 = 0;
        auto v = node_data.node_voltages[node_data.nodeindices[id]];
        inst.c2[id] = inst.c2[id] + 2.0;
        return ret_increment_c2;
    }


    inline double one_NetReceiveCalls(_nrn_mechanism_cache_range& _lmc, NetReceiveCalls_Instance& inst, NetReceiveCalls_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt) {
        double ret_one = 0.0;
        auto v = node_data.node_voltages[node_data.nodeindices[id]];
        ret_one = 1.0;
        return ret_one;
    }


    static void nrn_init_NetReceiveCalls(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_NetReceiveCalls(_lmc);
        auto node_data = make_node_data_NetReceiveCalls(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
            auto* _ppvar = _ml_arg->pdata[id];
            int node_id = node_data.nodeindices[id];
            auto v = node_data.node_voltages[node_id];
            inst.c1[id] = 0.0;
            inst.c2[id] = 0.0;
        }
    }


    static void nrn_jacob_NetReceiveCalls(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_NetReceiveCalls(_lmc);
        auto node_data = make_node_data_NetReceiveCalls(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
        }
    }
    static void nrn_net_receive_NetReceiveCalls(Point_process* _pnt, double* _args, double flag) {
        _nrn_mechanism_cache_instance _lmc{_pnt->prop};
        auto * nt = static_cast<NrnThread*>(_pnt->_vnt);
        auto * _ppvar = _nrn_mechanism_access_dparam(_pnt->prop);
        auto inst = make_instance_NetReceiveCalls(_lmc);
        auto node_data = make_node_data_NetReceiveCalls(_pnt->prop);
        // nocmodl has a nullptr dereference for thread variables.
        // NMODL will fail to compile at a later point, because of
        // missing '_thread_vars'.
        Datum * _thread = nullptr;
        size_t id = 0;
        double t = nt->_t;
        inst.c1[id] = inst.c1[id] + one_NetReceiveCalls(_lmc, inst, node_data, id, _ppvar, _thread, nt);
        increment_c2_NetReceiveCalls(_lmc, inst, node_data, id, _ppvar, _thread, nt);

    }
    static void nrn_destructor_NetReceiveCalls(Prop* prop) {
        Datum* _ppvar = _nrn_mechanism_access_dparam(prop);
        _nrn_mechanism_cache_instance _lmc{prop};
        const size_t id = 0;
        auto inst = make_instance_NetReceiveCalls(_lmc);
        auto node_data = make_node_data_NetReceiveCalls(prop);

    }


    static void _initlists() {
    }


    /** register channel with the simulator */
    extern "C" void _NetReceiveCalls_reg() {
        _initlists();

        _pointtype = point_register_mech(mechanism_info, nrn_alloc_NetReceiveCalls, nullptr, nullptr, nullptr, nrn_init_NetReceiveCalls, -1, 1, _hoc_create_pnt, _hoc_destroy_pnt, _member_func);

        mech_type = nrn_get_mechtype(mechanism_info[1]);
        hoc_register_parm_default(mech_type, &_parameter_defaults);
        _nrn_mechanism_register_data_fields(mech_type,
            _nrn_mechanism_field<double>{"c1"} /* 0 */,
            _nrn_mechanism_field<double>{"c2"} /* 1 */,
            _nrn_mechanism_field<double>{"v_unused"} /* 2 */,
            _nrn_mechanism_field<double>{"tsave"} /* 3 */,
            _nrn_mechanism_field<double*>{"node_area", "area"} /* 0 */,
            _nrn_mechanism_field<Point_process*>{"point_process", "pntproc"} /* 1 */
        );

        hoc_register_prop_size(mech_type, 4, 2);
        hoc_register_dparam_semantics(mech_type, 0, "area");
        hoc_register_dparam_semantics(mech_type, 1, "pntproc");
        hoc_register_var(hoc_scalar_double, hoc_vector_double, hoc_intfunc);
        pnt_receive[mech_type] = nrn_net_receive_NetReceiveCalls;
        pnt_receive_size[mech_type] = 1;
    }
}
