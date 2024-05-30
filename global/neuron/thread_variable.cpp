/*********************************************************
Model Name      : shared_global
Filename        : thread_variable.mod
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

static constexpr auto number_of_datum_variables = 0;
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

Prop* hoc_getdata_range(int type);


namespace neuron {
    #ifndef NRN_PRCELLSTATE
    #define NRN_PRCELLSTATE 0
    #endif


    /** channel information */
    static const char *mechanism_info[] = {
        "7.7.0",
        "shared_global",
        0,
        "y_shared_global",
        "z_shared_global",
        "il_shared_global",
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
    struct shared_global_Store {
        int thread_data_in_use{};
        double thread_data[1] /* TODO init thread_data */;
        double ggro{0};
        double ggp{9};
    };
    static_assert(std::is_trivially_copy_constructible_v<shared_global_Store>);
    static_assert(std::is_trivially_move_constructible_v<shared_global_Store>);
    static_assert(std::is_trivially_copy_assignable_v<shared_global_Store>);
    static_assert(std::is_trivially_move_assignable_v<shared_global_Store>);
    static_assert(std::is_trivially_destructible_v<shared_global_Store>);
    shared_global_Store shared_global_global;


    /** all mechanism instance variables and global variables */
    struct shared_global_Instance  {
        double* y{};
        double* z{};
        double* il{};
        double* v_unused{};
        double* g_unused{};
        shared_global_Store* global{&shared_global_global};
    };


    struct shared_global_NodeData  {
        int const * nodeindices;
        double const * node_voltages;
        double * node_diagonal;
        double * node_rhs;
        int nodecount;
    };


    static shared_global_Instance make_instance_shared_global(_nrn_mechanism_cache_range& _ml) {
        return shared_global_Instance {
            _ml.template fpfield_ptr<0>(),
            _ml.template fpfield_ptr<1>(),
            _ml.template fpfield_ptr<2>(),
            _ml.template fpfield_ptr<3>(),
            _ml.template fpfield_ptr<4>()
        };
    }


    static shared_global_NodeData make_node_data_shared_global(NrnThread& _nt, Memb_list& _ml_arg) {
        return shared_global_NodeData {
            _ml_arg.nodeindices,
            _nt.node_voltage_storage(),
            _nt.node_d_storage(),
            _nt.node_rhs_storage(),
            _ml_arg.nodecount
        };
    }


    static void nrn_alloc_shared_global(Prop* _prop) {
        Prop *prop_ion{};
        Datum *_ppvar{};
        _nrn_mechanism_cache_instance _ml_real{_prop};
        auto* const _ml = &_ml_real;
        size_t const _iml{};
        assert(_nrn_mechanism_get_num_vars(_prop) == 5);
        /*initialize range parameters*/
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


    /** connect global (scalar) variables to hoc -- */
    static DoubScal hoc_scalar_double[] = {
        {"ggro_shared_global", &shared_global_global.ggro},
        {"ggp_shared_global", &shared_global_global.ggp},
        {"ggw_shared_global", &shared_global_global.thread_data[0]},
        {nullptr, nullptr}
    };


    /** connect global (array) variables to hoc -- */
    static DoubVec hoc_vector_double[] = {
        {nullptr, nullptr, 0}
    };


    /* declaration of user functions */


    /* connect user functions to hoc names */
    static VoidFunc hoc_intfunc[] = {
        {"setdata_shared_global", _hoc_setdata},
        {0, 0}
    };
    static NPyDirectMechFunc npy_direct_func_proc[] = {
    };
    static void thread_mem_init(Datum* _thread)  {
        if(shared_global_global.thread_data_in_use) {
            std::cout << "thread_mem_init :: true " << std::endl;
            _thread[0] = {neuron::container::do_not_search, new double[1]{}};
        }
        else {
            std::cout << "thread_mem_init :: false " << std::endl;
            _thread[0] = {neuron::container::do_not_search, shared_global_global.thread_data};
            shared_global_global.thread_data_in_use = 1;
        }
    }
    static void thread_mem_cleanup(Datum* _thread)  {
        double * _thread_data_ptr = _thread[0].get<double*>();
        if(_thread_data_ptr == shared_global_global.thread_data) {
            std::cout << "thread_mem_cleanup :: true" << std::endl;
            shared_global_global.thread_data_in_use = 0;
        }
        else {
            std::cout << "thread_mem_cleanup :: false " << std::endl;
            delete[] _thread_data_ptr;
        }
    }


    void nrn_init_shared_global(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
        auto inst = make_instance_shared_global(_lmr);
        auto node_data = make_node_data_shared_global(*_nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* const _ml = &_lmr;
        auto* _thread = _ml_arg->_thread;
        double * _thread_globals = _thread[0].get<double*>();
        for (int id = 0; id < nodecount; id++) {
            auto* _ppvar = _ml_arg->pdata[id];
            int node_id = node_data.nodeindices[id];
            auto v = node_data.node_voltages[node_id];
            inst.v_unused[id] = v;
            printf("INITIAL %g\n", inst.z[id]);
            _thread_globals[0 + 0*1] = 48.0;
            inst.y[id] = 10.0;
        }
    }


    inline double nrn_current_shared_global(_nrn_mechanism_cache_range* _ml, NrnThread* _nt, Datum* _ppvar, Datum* _thread, double* _thread_globals, size_t id, shared_global_Instance& inst, shared_global_NodeData& node_data, double v) {
        double current = 0.0;
        if (_nt->_t > 0.33) {
            _thread_globals[0 + 0*1] = inst.global->ggp;
        }
        if (_nt->_t > 0.66) {
            _thread_globals[0 + 0*1] = inst.z[id];
        }
        inst.y[id] = _thread_globals[0 + 0*1];
        inst.il[id] = 0.0000001 * (v - 10.0);
        current += inst.il[id];
        return current;
    }


    /** update current */
    void nrn_cur_shared_global(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
        auto inst = make_instance_shared_global(_lmr);
        auto node_data = make_node_data_shared_global(*_nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* const _ml = &_lmr;
        auto* _thread = _ml_arg->_thread;
        double * _thread_globals = _thread[0].get<double*>();
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_data.nodeindices[id];
            double v = node_data.node_voltages[node_id];
            auto* _ppvar = _ml_arg->pdata[id];
            double I1 = nrn_current_shared_global(_ml, _nt, _ppvar, _thread, _thread_globals, id, inst, node_data, v+0.001);
            double I0 = nrn_current_shared_global(_ml, _nt, _ppvar, _thread, _thread_globals, id, inst, node_data, v);
            double rhs = I0;
            double g = (I1-I0)/0.001;
            node_data.node_rhs[node_id] -= rhs;
            // remember the conductances so we can set them later
            inst.g_unused[id] = g;
        }
    }


    void nrn_state_shared_global(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
        auto inst = make_instance_shared_global(_lmr);
        auto node_data = make_node_data_shared_global(*_nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* const _ml = &_lmr;
        auto* _thread = _ml_arg->_thread;
        double * _thread_globals = _thread[0].get<double*>();
        for (int id = 0; id < nodecount; id++) {
            
            int node_id = node_data.nodeindices[id];
            auto* _ppvar = _ml_arg->pdata[id];
            auto v = node_data.node_voltages[node_id];
        }
    }


    static void nrn_jacob_shared_global(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
        auto inst = make_instance_shared_global(_lmr);
        auto node_data = make_node_data_shared_global(*_nt, *_ml_arg);
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
    extern "C" void _thread_variable_reg() {
        _initlists();



        register_mech(mechanism_info, nrn_alloc_shared_global, nrn_cur_shared_global, nrn_jacob_shared_global, nrn_state_shared_global, nrn_init_shared_global, hoc_nrnpointerindex, 2);

        mech_type = nrn_get_mechtype(mechanism_info[1]);
        _nrn_mechanism_register_data_fields(mech_type,
            _nrn_mechanism_field<double>{"y"} /* 0 */,
            _nrn_mechanism_field<double>{"z"} /* 1 */,
            _nrn_mechanism_field<double>{"il"} /* 2 */,
            _nrn_mechanism_field<double>{"v_unused"} /* 3 */,
            _nrn_mechanism_field<double>{"g_unused"} /* 4 */
        );

        hoc_register_prop_size(mech_type, 5, 0);
        hoc_register_var(hoc_scalar_double, hoc_vector_double, hoc_intfunc);
        hoc_register_npy_direct(mech_type, npy_direct_func_proc);
        _nrn_thread_reg(mech_type, 1, thread_mem_init);
        _nrn_thread_reg(mech_type, 0, thread_mem_cleanup);
    }
}
