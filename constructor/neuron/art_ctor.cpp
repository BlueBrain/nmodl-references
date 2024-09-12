/*********************************************************
Model Name      : art_ctor
Filename        : art_ctor.mod
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
extern void _nrn_thread_reg(int, int, void(*)(Datum*));

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
extern Node* nrn_alloc_node_;


namespace neuron {
    #ifndef NRN_PRCELLSTATE
    #define NRN_PRCELLSTATE 0
    #endif


    /** channel information */
    static const char *mechanism_info[] = {
        "7.7.0",
        "art_ctor",
        0,
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
    struct art_ctor_Store {
        double thread_data_in_use{0};
        double thread_data[2] /* TODO init const-array */;
    };
    static_assert(std::is_trivially_copy_constructible_v<art_ctor_Store>);
    static_assert(std::is_trivially_move_constructible_v<art_ctor_Store>);
    static_assert(std::is_trivially_copy_assignable_v<art_ctor_Store>);
    static_assert(std::is_trivially_move_assignable_v<art_ctor_Store>);
    static_assert(std::is_trivially_destructible_v<art_ctor_Store>);
    art_ctor_Store art_ctor_global;
    static std::vector<double> _parameter_defaults = {
    };


    /** all mechanism instance variables and global variables */
    struct art_ctor_Instance  {
        double* v_unused{};
        const double* const* node_area{};
        art_ctor_Store* global{&art_ctor_global};
    };


    struct art_ctor_NodeData  {
        int const * nodeindices;
        double const * node_voltages;
        double * node_diagonal;
        double * node_rhs;
        int nodecount;
    };


    struct art_ctor_ThreadVariables  {
        double * thread_data;

        double * ctor_calls_ptr(size_t id) {
            return thread_data + 0 + (id % 1);
        }
        double & ctor_calls(size_t id) {
            return thread_data[0 + (id % 1)];
        }
        double * dtor_calls_ptr(size_t id) {
            return thread_data + 1 + (id % 1);
        }
        double & dtor_calls(size_t id) {
            return thread_data[1 + (id % 1)];
        }

        art_ctor_ThreadVariables(double * const thread_data) {
            this->thread_data = thread_data;
        }
    };


    static art_ctor_Instance make_instance_art_ctor(_nrn_mechanism_cache_range& _lmc) {
        return art_ctor_Instance {
            _lmc.template fpfield_ptr<0>(),
            _lmc.template dptr_field_ptr<0>()
        };
    }


    static art_ctor_NodeData make_node_data_art_ctor(NrnThread& nt, Memb_list& _ml_arg) {
        return art_ctor_NodeData {
            _ml_arg.nodeindices,
            nt.node_voltage_storage(),
            nt.node_d_storage(),
            nt.node_rhs_storage(),
            _ml_arg.nodecount
        };
    }
    static art_ctor_NodeData make_node_data_art_ctor(Prop * _prop) {
        static std::vector<int> node_index{0};
        Node* _node = _nrn_mechanism_access_node(_prop);
        return art_ctor_NodeData {
            node_index.data(),
            &_nrn_mechanism_access_voltage(_node),
            &_nrn_mechanism_access_d(_node),
            &_nrn_mechanism_access_rhs(_node),
            1
        };
    }

    void nrn_constructor_art_ctor(Prop* prop);
    void nrn_destructor_art_ctor(Prop* prop);


    static void nrn_alloc_art_ctor(Prop* _prop) {
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
            nrn_constructor_art_ctor(_prop);
        }
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
        {"ctor_calls_art_ctor", &art_ctor_global.thread_data[0]},
        {"dtor_calls_art_ctor", &art_ctor_global.thread_data[1]},
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
    static void thread_mem_init(Datum* _thread)  {
        if(art_ctor_global.thread_data_in_use) {
            _thread[0] = {neuron::container::do_not_search, new double[2]{}};
        }
        else {
            _thread[0] = {neuron::container::do_not_search, art_ctor_global.thread_data};
            art_ctor_global.thread_data_in_use = 1;
        }
    }
    static void thread_mem_cleanup(Datum* _thread)  {
        double * _thread_data_ptr = _thread[0].get<double*>();
        if(_thread_data_ptr == art_ctor_global.thread_data) {
            art_ctor_global.thread_data_in_use = 0;
        }
        else {
            delete[] _thread_data_ptr;
        }
    }


    void nrn_init_art_ctor(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _type};
        auto inst = make_instance_art_ctor(_lmc);
        auto node_data = make_node_data_art_ctor(*nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* _thread = _ml_arg->_thread;
        auto _thread_vars = art_ctor_ThreadVariables(_thread[0].get<double*>());
        for (int id = 0; id < nodecount; id++) {
            auto* _ppvar = _ml_arg->pdata[id];
        }
    }


    static void nrn_jacob_art_ctor(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _type};
        auto inst = make_instance_art_ctor(_lmc);
        auto node_data = make_node_data_art_ctor(*nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
        }
    }
    void nrn_constructor_art_ctor(Prop* prop) {
        Datum* _ppvar = _nrn_mechanism_access_dparam(prop);
        _nrn_mechanism_cache_instance _lmc{prop};
        const size_t id = 0;
        auto inst = make_instance_art_ctor(_lmc);
        auto _thread_vars = art_ctor_ThreadVariables(art_ctor_global.thread_data);

        _thread_vars.ctor_calls(id) = _thread_vars.ctor_calls(id) + 1.0;
    }
    void nrn_destructor_art_ctor(Prop* prop) {
        Datum* _ppvar = _nrn_mechanism_access_dparam(prop);
        _nrn_mechanism_cache_instance _lmc{prop};
        const size_t id = 0;
        auto inst = make_instance_art_ctor(_lmc);
        auto _thread_vars = art_ctor_ThreadVariables(art_ctor_global.thread_data);

        _thread_vars.dtor_calls(id) = _thread_vars.dtor_calls(id) + 1.0;
    }


    static void _initlists() {
    }


    /** register channel with the simulator */
    extern "C" void _art_ctor_reg() {
        _initlists();

        _pointtype = point_register_mech(mechanism_info, nrn_alloc_art_ctor, nullptr, nullptr, nullptr, nrn_init_art_ctor, hoc_nrnpointerindex, 2, _hoc_create_pnt, _hoc_destroy_pnt, _member_func);
        register_destructor(nrn_destructor_art_ctor);
        _extcall_thread.resize(2);
        thread_mem_init(_extcall_thread.data());
        art_ctor_global.thread_data_in_use = 0;

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
        _nrn_thread_reg(mech_type, 1, thread_mem_init);
        _nrn_thread_reg(mech_type, 0, thread_mem_cleanup);
    }
}
