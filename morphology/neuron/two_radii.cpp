/*********************************************************
Model Name      : two_radii
Filename        : two_radii.mod
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

Prop* hoc_getdata_range(int type);
extern Node* nrn_alloc_node_;


namespace neuron {
    #ifndef NRN_PRCELLSTATE
    #define NRN_PRCELLSTATE 0
    #endif


    /** channel information */
    static const char *mechanism_info[] = {
        "7.7.0",
        "two_radii",
        0,
        "il_two_radii",
        "inv_two_radii",
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
    struct two_radii_Store {
        Symbol* _morphology_sym;
    };
    static_assert(std::is_trivially_copy_constructible_v<two_radii_Store>);
    static_assert(std::is_trivially_move_constructible_v<two_radii_Store>);
    static_assert(std::is_trivially_copy_assignable_v<two_radii_Store>);
    static_assert(std::is_trivially_move_assignable_v<two_radii_Store>);
    static_assert(std::is_trivially_destructible_v<two_radii_Store>);
    two_radii_Store two_radii_global;
    static std::vector<double> _parameter_defaults = {
    };


    /** all mechanism instance variables and global variables */
    struct two_radii_Instance  {
        double* il{};
        double* inv{};
        double* v_unused{};
        double* g_unused{};
        double* const* diam{};
        double* const* area{};
        two_radii_Store* global{&two_radii_global};
    };


    struct two_radii_NodeData  {
        int const * nodeindices;
        double const * node_voltages;
        double * node_diagonal;
        double * node_rhs;
        int nodecount;
    };


    static two_radii_Instance make_instance_two_radii(_nrn_mechanism_cache_range& _lmc) {
        return two_radii_Instance {
            _lmc.template fpfield_ptr<0>(),
            _lmc.template fpfield_ptr<1>(),
            _lmc.template fpfield_ptr<2>(),
            _lmc.template fpfield_ptr<3>(),
            _lmc.template dptr_field_ptr<0>(),
            _lmc.template dptr_field_ptr<1>()
        };
    }


    static two_radii_NodeData make_node_data_two_radii(NrnThread& nt, Memb_list& _ml_arg) {
        return two_radii_NodeData {
            _ml_arg.nodeindices,
            nt.node_voltage_storage(),
            nt.node_d_storage(),
            nt.node_rhs_storage(),
            _ml_arg.nodecount
        };
    }
    static two_radii_NodeData make_node_data_two_radii(Prop * _prop) {
        static std::vector<int> node_index{0};
        Node* _node = _nrn_mechanism_access_node(_prop);
        return two_radii_NodeData {
            node_index.data(),
            &_nrn_mechanism_access_voltage(_node),
            &_nrn_mechanism_access_d(_node),
            &_nrn_mechanism_access_rhs(_node),
            1
        };
    }

    void nrn_destructor_two_radii(Prop* prop) {
        Datum* _ppvar = _nrn_mechanism_access_dparam(prop);
        _nrn_mechanism_cache_instance _lmc{prop};
        const size_t id = 0;
        auto inst = make_instance_two_radii(_lmc);
        auto node_data = make_node_data_two_radii(prop);

    }


    static void nrn_alloc_two_radii(Prop* _prop) {
        Datum *_ppvar = nullptr;
        _ppvar = nrn_prop_datum_alloc(mech_type, 2, _prop);
        _nrn_mechanism_access_dparam(_prop) = _ppvar;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const _iml = 0;
        assert(_nrn_mechanism_get_num_vars(_prop) == 4);
        /*initialize range parameters*/
        _nrn_mechanism_access_dparam(_prop) = _ppvar;
        Prop * morphology_prop = need_memb(two_radii_global._morphology_sym);
        _ppvar[0] = _nrn_mechanism_get_param_handle(morphology_prop, 0);
        _ppvar[1] = _nrn_mechanism_get_area_handle(nrn_alloc_node_);
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
    inline double square_diam_two_radii(_nrn_mechanism_cache_range& _lmc, two_radii_Instance& inst, two_radii_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt);
    inline double square_area_two_radii(_nrn_mechanism_cache_range& _lmc, two_radii_Instance& inst, two_radii_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt);


    /** connect global (scalar) variables to hoc -- */
    static DoubScal hoc_scalar_double[] = {
        {nullptr, nullptr}
    };


    /** connect global (array) variables to hoc -- */
    static DoubVec hoc_vector_double[] = {
        {nullptr, nullptr, 0}
    };


    /* declaration of user functions */
    static void _hoc_square_diam(void);
    static void _hoc_square_area(void);
    static double _npy_square_diam(Prop*);
    static double _npy_square_area(Prop*);


    /* connect user functions to hoc names */
    static VoidFunc hoc_intfunc[] = {
        {"setdata_two_radii", _hoc_setdata},
        {"square_diam_two_radii", _hoc_square_diam},
        {"square_area_two_radii", _hoc_square_area},
        {nullptr, nullptr}
    };
    static NPyDirectMechFunc npy_direct_func_proc[] = {
        {"square_diam", _npy_square_diam},
        {"square_area", _npy_square_area},
        {nullptr, nullptr}
    };
    static void _hoc_square_diam(void) {
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
        auto inst = make_instance_two_radii(_lmc);
        auto node_data = make_node_data_two_radii(_local_prop);
        _r = square_diam_two_radii(_lmc, inst, node_data, id, _ppvar, _thread, nt);
        hoc_retpushx(_r);
    }
    static double _npy_square_diam(Prop* _prop) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id = 0;
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_two_radii(_lmc);
        auto node_data = make_node_data_two_radii(_prop);
        _r = square_diam_two_radii(_lmc, inst, node_data, id, _ppvar, _thread, nt);
        return(_r);
    }
    static void _hoc_square_area(void) {
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
        auto inst = make_instance_two_radii(_lmc);
        auto node_data = make_node_data_two_radii(_local_prop);
        _r = square_area_two_radii(_lmc, inst, node_data, id, _ppvar, _thread, nt);
        hoc_retpushx(_r);
    }
    static double _npy_square_area(Prop* _prop) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id = 0;
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_two_radii(_lmc);
        auto node_data = make_node_data_two_radii(_prop);
        _r = square_area_two_radii(_lmc, inst, node_data, id, _ppvar, _thread, nt);
        return(_r);
    }


    inline double square_diam_two_radii(_nrn_mechanism_cache_range& _lmc, two_radii_Instance& inst, two_radii_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt) {
        double ret_square_diam = 0.0;
        auto v = node_data.node_voltages[node_data.nodeindices[id]];
        ret_square_diam = (*inst.diam[id]) * (*inst.diam[id]);
        return ret_square_diam;
    }


    inline double square_area_two_radii(_nrn_mechanism_cache_range& _lmc, two_radii_Instance& inst, two_radii_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt) {
        double ret_square_area = 0.0;
        auto v = node_data.node_voltages[node_data.nodeindices[id]];
        ret_square_area = (*inst.area[id]) * (*inst.area[id]);
        return ret_square_area;
    }


    void nrn_init_two_radii(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _type};
        auto inst = make_instance_two_radii(_lmc);
        auto node_data = make_node_data_two_radii(*nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* _thread = _ml_arg->_thread;
        for (int id = 0; id < nodecount; id++) {
            auto* _ppvar = _ml_arg->pdata[id];
            int node_id = node_data.nodeindices[id];
            auto v = node_data.node_voltages[node_id];
            inst.inv[id] = 1.0 / (square_diam_two_radii(_lmc, inst, node_data, id, _ppvar, _thread, nt) + (*inst.area[id]));
        }
    }


    inline double nrn_current_two_radii(_nrn_mechanism_cache_range& _lmc, NrnThread* nt, Datum* _ppvar, Datum* _thread, size_t id, two_radii_Instance& inst, two_radii_NodeData& node_data, double v) {
        double current = 0.0;
        inst.il[id] = (square_diam_two_radii(_lmc, inst, node_data, id, _ppvar, _thread, nt) + (*inst.area[id])) * 0.001 * (v - 20.0);
        current += inst.il[id];
        return current;
    }


    /** update current */
    void nrn_cur_two_radii(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _type};
        auto inst = make_instance_two_radii(_lmc);
        auto node_data = make_node_data_two_radii(*nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* _thread = _ml_arg->_thread;
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_data.nodeindices[id];
            double v = node_data.node_voltages[node_id];
            auto* _ppvar = _ml_arg->pdata[id];
            double I1 = nrn_current_two_radii(_lmc, nt, _ppvar, _thread, id, inst, node_data, v+0.001);
            double I0 = nrn_current_two_radii(_lmc, nt, _ppvar, _thread, id, inst, node_data, v);
            double rhs = I0;
            double g = (I1-I0)/0.001;
            node_data.node_rhs[node_id] -= rhs;
            inst.g_unused[id] = g;
        }
    }


    void nrn_state_two_radii(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _type};
        auto inst = make_instance_two_radii(_lmc);
        auto node_data = make_node_data_two_radii(*nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* _thread = _ml_arg->_thread;
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_data.nodeindices[id];
            auto* _ppvar = _ml_arg->pdata[id];
            auto v = node_data.node_voltages[node_id];
        }
    }


    static void nrn_jacob_two_radii(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _type};
        auto inst = make_instance_two_radii(_lmc);
        auto node_data = make_node_data_two_radii(*nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_data.nodeindices[id];
            node_data.node_diagonal[node_id] += inst.g_unused[id];
        }
    }


    static void _initlists() {
    }


    /** register channel with the simulator */
    extern "C" void _two_radii_reg() {
        _initlists();

        register_mech(mechanism_info, nrn_alloc_two_radii, nrn_cur_two_radii, nrn_jacob_two_radii, nrn_state_two_radii, nrn_init_two_radii, hoc_nrnpointerindex, 1);

        mech_type = nrn_get_mechtype(mechanism_info[1]);
        hoc_register_parm_default(mech_type, &_parameter_defaults);
        _nrn_mechanism_register_data_fields(mech_type,
            _nrn_mechanism_field<double>{"il"} /* 0 */,
            _nrn_mechanism_field<double>{"inv"} /* 1 */,
            _nrn_mechanism_field<double>{"v_unused"} /* 2 */,
            _nrn_mechanism_field<double>{"g_unused"} /* 3 */,
            _nrn_mechanism_field<double*>{"diam", "diam"} /* 0 */,
            _nrn_mechanism_field<double*>{"area", "area"} /* 1 */
        );

        hoc_register_prop_size(mech_type, 4, 2);
        hoc_register_dparam_semantics(mech_type, 0, "diam");
        hoc_register_dparam_semantics(mech_type, 1, "area");
        hoc_register_var(hoc_scalar_double, hoc_vector_double, hoc_intfunc);
        hoc_register_npy_direct(mech_type, npy_direct_func_proc);
        two_radii_global._morphology_sym = hoc_lookup("morphology");
    }
}
