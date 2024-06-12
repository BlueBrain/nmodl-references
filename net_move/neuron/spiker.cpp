/*********************************************************
Model Name      : spiker
Filename        : spiker.mod
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

static constexpr auto number_of_datum_variables = 3;
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


namespace neuron {
    #ifndef NRN_PRCELLSTATE
    #define NRN_PRCELLSTATE 0
    #endif


    /** channel information */
    static const char *mechanism_info[] = {
        "7.7.0",
        "spiker",
        0,
        "y",
        "z",
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
    struct spiker_Store {
    };
    static_assert(std::is_trivially_copy_constructible_v<spiker_Store>);
    static_assert(std::is_trivially_move_constructible_v<spiker_Store>);
    static_assert(std::is_trivially_copy_assignable_v<spiker_Store>);
    static_assert(std::is_trivially_move_assignable_v<spiker_Store>);
    static_assert(std::is_trivially_destructible_v<spiker_Store>);
    spiker_Store spiker_global;


    /** all mechanism instance variables and global variables */
    struct spiker_Instance  {
        double* y{};
        double* z{};
        double* v_unused{};
        double* tsave{};
        const double* const* node_area{};
        const int* const* tqitem{};
        spiker_Store* global{&spiker_global};
    };


    struct spiker_NodeData  {
        int const * nodeindices;
        double const * node_voltages;
        double * node_diagonal;
        double * node_rhs;
        int nodecount;
    };


    static spiker_Instance make_instance_spiker(_nrn_mechanism_cache_range& _lmc) {
        return spiker_Instance {
            _lmc.template fpfield_ptr<0>(),
            _lmc.template fpfield_ptr<1>(),
            _lmc.template fpfield_ptr<2>(),
            _lmc.template fpfield_ptr<3>(),
            _lmc.template dptr_field_ptr<0>()
        };
    }


    static spiker_NodeData make_node_data_spiker(NrnThread& _nt, Memb_list& _ml_arg) {
        return spiker_NodeData {
            _ml_arg.nodeindices,
            _nt.node_voltage_storage(),
            _nt.node_d_storage(),
            _nt.node_rhs_storage(),
            _ml_arg.nodecount
        };
    }


    static void nrn_alloc_spiker(Prop* _prop) {
        Prop *prop_ion{};
        Datum *_ppvar{};
        if (nrn_point_prop_) {
            _nrn_mechanism_access_alloc_seq(_prop) = _nrn_mechanism_access_alloc_seq(nrn_point_prop_);
            _ppvar = _nrn_mechanism_access_dparam(nrn_point_prop_);
        } else {
            _ppvar = nrn_prop_datum_alloc(mech_type, 3, _prop);
            _nrn_mechanism_access_dparam(_prop) = _ppvar;
            _nrn_mechanism_cache_instance _lmc{_prop};
            size_t const _iml{};
            assert(_nrn_mechanism_get_num_vars(_prop) == 4);
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


    void nrn_init_spiker(const _nrn_model_sorted_token& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *_nt, *_ml_arg, _type};
        auto inst = make_instance_spiker(_lmc);
        auto node_data = make_node_data_spiker(*_nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* _thread = _ml_arg->_thread;
        for (int id = 0; id < nodecount; id++) {
            auto* _ppvar = _ml_arg->pdata[id];
            int node_id = node_data.nodeindices[id];
            auto v = node_data.node_voltages[node_id];
            inst.v_unused[id] = v;
            inst.y[id] = 0.0;
            inst.z[id] = 0.0;
            net_send(/* tqitem */ &_ppvar[2], nullptr, _ppvar[1].get<Point_process*>(), _nt->_t + 1.8, 1.0);
        }
    }


    static void nrn_jacob_spiker(const _nrn_model_sorted_token& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *_nt, *_ml_arg, _type};
        auto inst = make_instance_spiker(_lmc);
        auto node_data = make_node_data_spiker(*_nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
        }
    }
    static void nrn_net_receive_spiker(Point_process* _pnt, double* _args, double flag) {
        _nrn_mechanism_cache_instance _lmc{_pnt->prop};
        auto * _nt = static_cast<NrnThread*>(_pnt->_vnt);
        auto * _ppvar = _nrn_mechanism_access_dparam(_pnt->prop);
        auto inst = make_instance_spiker(_lmc);
        size_t id = 0;
        double t = _nt->_t;
        if (flag == 0.0) {
            inst.y[id] = inst.y[id] + 1.0;
            net_move(/* tqitem */ &_ppvar[2], _pnt, _nt->_t + 0.1);
        } else {
            inst.z[id] = inst.z[id] + 1.0;
            net_send(/* tqitem */ &_ppvar[2], nullptr, _pnt, _nt->_t + 2.0, 1.0);
        }

    }


    static void _initlists() {
    }


    /** register channel with the simulator */
    extern "C" void _spiker_reg() {
        _initlists();

        _pointtype = point_register_mech(mechanism_info, nrn_alloc_spiker, nullptr, nullptr, nullptr, nrn_init_spiker, hoc_nrnpointerindex, 1, _hoc_create_pnt, _hoc_destroy_pnt, _member_func);

        mech_type = nrn_get_mechtype(mechanism_info[1]);
        _nrn_mechanism_register_data_fields(mech_type,
            _nrn_mechanism_field<double>{"y"} /* 0 */,
            _nrn_mechanism_field<double>{"z"} /* 1 */,
            _nrn_mechanism_field<double>{"v_unused"} /* 2 */,
            _nrn_mechanism_field<double>{"tsave"} /* 3 */,
            _nrn_mechanism_field<double*>{"node_area", "area"} /* 0 */,
            _nrn_mechanism_field<Point_process*>{"point_process", "pntproc"} /* 1 */,
            _nrn_mechanism_field<void*>{"tqitem", "netsend"} /* 2 */
        );

        hoc_register_prop_size(mech_type, 4, 3);
        hoc_register_dparam_semantics(mech_type, 0, "area");
        hoc_register_dparam_semantics(mech_type, 1, "pntproc");
        hoc_register_dparam_semantics(mech_type, 2, "netsend");
        hoc_register_var(hoc_scalar_double, hoc_vector_double, hoc_intfunc);
        pnt_receive[mech_type] = nrn_net_receive_spiker;
        pnt_receive_size[mech_type] = 1;
    }
}
