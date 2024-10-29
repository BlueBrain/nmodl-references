/*********************************************************
Model Name      : ForNetconsSyn
Filename        : for_netcons_syn.mod
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

static constexpr auto number_of_datum_variables = 3;
static constexpr auto number_of_floating_point_variables = 3;

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
int _nrn_netcon_args(void*, double***);
extern Node* nrn_alloc_node_;


namespace neuron {
    #ifndef NRN_PRCELLSTATE
    #define NRN_PRCELLSTATE 0
    #endif


    /** channel information */
    static const char *mechanism_info[] = {
        "7.7.0",
        "ForNetconsSyn",
        0,
        "a0",
        0,
        0,
        0
    };


    /* NEURON global variables */
    static int mech_type;
    static int _pointtype;
    static _nrn_mechanism_std_vector<Datum> _extcall_thread;


    /** all global variables */
    struct ForNetconsSyn_Store {
    };
    static_assert(std::is_trivially_copy_constructible_v<ForNetconsSyn_Store>);
    static_assert(std::is_trivially_move_constructible_v<ForNetconsSyn_Store>);
    static_assert(std::is_trivially_copy_assignable_v<ForNetconsSyn_Store>);
    static_assert(std::is_trivially_move_assignable_v<ForNetconsSyn_Store>);
    static_assert(std::is_trivially_destructible_v<ForNetconsSyn_Store>);
    static ForNetconsSyn_Store ForNetconsSyn_global;
    static std::vector<double> _parameter_defaults = {
    };


    /** all mechanism instance variables and global variables */
    struct ForNetconsSyn_Instance  {
        double* a0{};
        double* v_unused{};
        double* tsave{};
        const double* const* node_area{};
        ForNetconsSyn_Store* global{&ForNetconsSyn_global};
    };


    struct ForNetconsSyn_NodeData  {
        int const * nodeindices;
        double const * node_voltages;
        double * node_diagonal;
        double * node_rhs;
        int nodecount;
    };


    static ForNetconsSyn_Instance make_instance_ForNetconsSyn(_nrn_mechanism_cache_range* _lmc) {
        if(_lmc == nullptr) {
            return ForNetconsSyn_Instance();
        }

        return ForNetconsSyn_Instance {
            _lmc->template fpfield_ptr<0>(),
            _lmc->template fpfield_ptr<1>(),
            _lmc->template fpfield_ptr<2>(),
            _lmc->template dptr_field_ptr<0>()
        };
    }


    static ForNetconsSyn_NodeData make_node_data_ForNetconsSyn(NrnThread& nt, Memb_list& _ml_arg) {
        return ForNetconsSyn_NodeData {
            _ml_arg.nodeindices,
            nt.node_voltage_storage(),
            nt.node_d_storage(),
            nt.node_rhs_storage(),
            _ml_arg.nodecount
        };
    }
    static ForNetconsSyn_NodeData make_node_data_ForNetconsSyn(Prop * _prop) {
        if(!_prop) {
            return ForNetconsSyn_NodeData();
        }

        static std::vector<int> node_index{0};
        Node* _node = _nrn_mechanism_access_node(_prop);
        return ForNetconsSyn_NodeData {
            node_index.data(),
            &_nrn_mechanism_access_voltage(_node),
            &_nrn_mechanism_access_d(_node),
            &_nrn_mechanism_access_rhs(_node),
            1
        };
    }

    static void nrn_destructor_ForNetconsSyn(Prop* prop);


    static void nrn_alloc_ForNetconsSyn(Prop* _prop) {
        Datum *_ppvar = nullptr;
        if (nrn_point_prop_) {
            _nrn_mechanism_access_alloc_seq(_prop) = _nrn_mechanism_access_alloc_seq(nrn_point_prop_);
            _ppvar = _nrn_mechanism_access_dparam(nrn_point_prop_);
        } else {
            _ppvar = nrn_prop_datum_alloc(mech_type, 3, _prop);
            _nrn_mechanism_access_dparam(_prop) = _ppvar;
            _nrn_mechanism_cache_instance _lmc{_prop};
            size_t const _iml = 0;
            assert(_nrn_mechanism_get_num_vars(_prop) == 3);
            /*initialize range parameters*/
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
        Prop* _prop = ((Point_process*)_vptr)->prop;
        if (_prop) {
            Datum* _ppvar = _nrn_mechanism_access_dparam(_prop);
            _nrn_free_fornetcon(&_ppvar[2].literal_value<void*>());
        }
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


    static void nrn_init_ForNetconsSyn(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_ForNetconsSyn(&_lmc);
        auto node_data = make_node_data_ForNetconsSyn(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
            auto* _ppvar = _ml_arg->pdata[id];
            int node_id = node_data.nodeindices[id];
            inst.v_unused[id] = node_data.node_voltages[node_id];
        }
    }


    static void nrn_jacob_ForNetconsSyn(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_ForNetconsSyn(&_lmc);
        auto node_data = make_node_data_ForNetconsSyn(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
        }
    }
    static void nrn_net_receive_ForNetconsSyn(Point_process* _pnt, double* _args, double flag) {
        _nrn_mechanism_cache_instance _lmc{_pnt->prop};
        auto * nt = static_cast<NrnThread*>(_pnt->_vnt);
        auto * _ppvar = _nrn_mechanism_access_dparam(_pnt->prop);
        auto inst = make_instance_ForNetconsSyn(&_lmc);
        auto node_data = make_node_data_ForNetconsSyn(_pnt->prop);
        // nocmodl has a nullptr dereference for thread variables.
        // NMODL will fail to compile at a later point, because of
        // missing '_thread_vars'.
        Datum * _thread = nullptr;
        size_t id = 0;
        double t = nt->_t;
        double ** _fornetcon_data;
        int _n_netcons = _nrn_netcon_args(_ppvar[2].literal_value<void*>(), &_fornetcon_data);
        for (size_t _i = 0; _i < _n_netcons; ++_i) {
            double * _netcon_data = _fornetcon_data[_i];
            _netcon_data[1] = 2.0 * _netcon_data[1];
        }


    }


    static void net_init(Point_process* _pnt, double* _args, double flag) {
        _nrn_mechanism_cache_instance _lmc{_pnt->prop};
        auto * nt = static_cast<NrnThread*>(_pnt->_vnt);
        auto * _ppvar = _nrn_mechanism_access_dparam(_pnt->prop);
        auto inst = make_instance_ForNetconsSyn(&_lmc);
        auto node_data = make_node_data_ForNetconsSyn(_pnt->prop);
        // nocmodl has a nullptr dereference for thread variables.
        // NMODL will fail to compile at a later point, because of
        // missing '_thread_vars'.
        Datum * _thread = nullptr;
        size_t id = 0;
        double t = nt->_t;
        _args[1] = inst.a0[id];
    }
    static void nrn_destructor_ForNetconsSyn(Prop* prop) {
        Datum* _ppvar = _nrn_mechanism_access_dparam(prop);
        _nrn_mechanism_cache_instance _lmc{prop};
        const size_t id = 0;
        auto inst = make_instance_ForNetconsSyn(prop ? &_lmc : nullptr);
        auto node_data = make_node_data_ForNetconsSyn(prop);

    }


    static void _initlists() {
    }


    extern "C" void _for_netcons_syn_reg() {
        _initlists();

        _pointtype = point_register_mech(mechanism_info, nrn_alloc_ForNetconsSyn, nullptr, nullptr, nullptr, nrn_init_ForNetconsSyn, -1, 1, _hoc_create_pnt, _hoc_destroy_pnt, _member_func);

        mech_type = nrn_get_mechtype(mechanism_info[1]);
        hoc_register_parm_default(mech_type, &_parameter_defaults);
        _nrn_mechanism_register_data_fields(mech_type,
            _nrn_mechanism_field<double>{"a0"} /* 0 */,
            _nrn_mechanism_field<double>{"v_unused"} /* 1 */,
            _nrn_mechanism_field<double>{"tsave"} /* 2 */,
            _nrn_mechanism_field<double*>{"node_area", "area"} /* 0 */,
            _nrn_mechanism_field<Point_process*>{"point_process", "pntproc"} /* 1 */,
            _nrn_mechanism_field<void*>{"fornetcon_data", "fornetcon"} /* 2 */
        );

        hoc_register_prop_size(mech_type, 3, 3);
        hoc_register_dparam_semantics(mech_type, 0, "area");
        hoc_register_dparam_semantics(mech_type, 1, "pntproc");
        hoc_register_dparam_semantics(mech_type, 2, "fornetcon");
        add_nrn_fornetcons(mech_type, 2);
        hoc_register_var(hoc_scalar_double, hoc_vector_double, hoc_intfunc);
        pnt_receive[mech_type] = nrn_net_receive_ForNetconsSyn;
        pnt_receive_size[mech_type] = 2;
        pnt_receive_init[mech_type] = net_init;
    }
}
