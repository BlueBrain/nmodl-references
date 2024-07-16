/*********************************************************
Model Name      : thread_newton
Filename        : thread_newton.mod
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
#include "solver/newton.hpp"

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
extern Node* nrn_alloc_node_;


namespace neuron {
    #ifndef NRN_PRCELLSTATE
    #define NRN_PRCELLSTATE 0
    #endif


    /** channel information */
    static const char *mechanism_info[] = {
        "7.7.0",
        "thread_newton",
        0,
        "x_thread_newton",
        0,
        "X_thread_newton",
        0,
        0
    };


    /* NEURON global variables */
    static neuron::container::field_index _slist1[1], _dlist1[1];
    static int mech_type;
    static Prop* _extcall_prop;
    /* _prop_id kind of shadows _extcall_prop to allow validity checking. */
    static _nrn_non_owning_id_without_container _prop_id{};
    static int hoc_nrnpointerindex = -1;
    static _nrn_mechanism_std_vector<Datum> _extcall_thread;


    /** all global variables */
    struct thread_newton_Store {
        int thread_data_in_use{};
        double thread_data[1];
        double X0{};
    };
    static_assert(std::is_trivially_copy_constructible_v<thread_newton_Store>);
    static_assert(std::is_trivially_move_constructible_v<thread_newton_Store>);
    static_assert(std::is_trivially_copy_assignable_v<thread_newton_Store>);
    static_assert(std::is_trivially_move_assignable_v<thread_newton_Store>);
    static_assert(std::is_trivially_destructible_v<thread_newton_Store>);
    thread_newton_Store thread_newton_global;


    /** all mechanism instance variables and global variables */
    struct thread_newton_Instance  {
        double* x{};
        double* X{};
        double* DX{};
        double* v_unused{};
        double* g_unused{};
        thread_newton_Store* global{&thread_newton_global};
    };


    struct thread_newton_NodeData  {
        int const * nodeindices;
        double const * node_voltages;
        double * node_diagonal;
        double * node_rhs;
        int nodecount;
    };


    struct thread_newton_ThreadVariables  {
        double * thread_data;

        double * c_ptr(size_t id) {
            return thread_data + 0 + (id % 1);
        }
        double & c(size_t id) {
            return thread_data[0 + (id % 1)];
        }

        thread_newton_ThreadVariables(double * const thread_data) {
            this->thread_data = thread_data;
        }
    };


    static thread_newton_Instance make_instance_thread_newton(_nrn_mechanism_cache_range& _lmc) {
        return thread_newton_Instance {
            _lmc.template fpfield_ptr<0>(),
            _lmc.template fpfield_ptr<1>(),
            _lmc.template fpfield_ptr<2>(),
            _lmc.template fpfield_ptr<3>(),
            _lmc.template fpfield_ptr<4>()
        };
    }


    static thread_newton_NodeData make_node_data_thread_newton(NrnThread& nt, Memb_list& _ml_arg) {
        return thread_newton_NodeData {
            _ml_arg.nodeindices,
            nt.node_voltage_storage(),
            nt.node_d_storage(),
            nt.node_rhs_storage(),
            _ml_arg.nodecount
        };
    }


    static void nrn_alloc_thread_newton(Prop* _prop) {
        Datum *_ppvar = nullptr;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const _iml = 0;
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


    struct functor_thread_newton_0 {
        _nrn_mechanism_cache_range& _lmc;
        thread_newton_Instance& inst;
        size_t id;
        Datum* _ppvar;
        Datum* _thread;
        thread_newton_ThreadVariables& _thread_vars;
        NrnThread* nt;
        double v;
        double source0_, old_X;

        void initialize() {
            source0_ = _thread_vars.c(id);
            old_X = inst.X[id];
        }

        functor_thread_newton_0(_nrn_mechanism_cache_range& _lmc, thread_newton_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, thread_newton_ThreadVariables& _thread_vars, NrnThread* nt, double v)
            : _lmc(_lmc), inst(inst), id(id), _ppvar(_ppvar), _thread(_thread), _thread_vars(_thread_vars), nt(nt), v(v)
        {}
        void operator()(const Eigen::Matrix<double, 1, 1>& nmodl_eigen_xm, Eigen::Matrix<double, 1, 1>& nmodl_eigen_fm, Eigen::Matrix<double, 1, 1>& nmodl_eigen_jm) const {
            const double* nmodl_eigen_x = nmodl_eigen_xm.data();
            double* nmodl_eigen_j = nmodl_eigen_jm.data();
            double* nmodl_eigen_f = nmodl_eigen_fm.data();
            nmodl_eigen_f[static_cast<int>(0)] =  -nmodl_eigen_x[static_cast<int>(0)] + nt->_dt * source0_ + old_X;
            nmodl_eigen_j[static_cast<int>(0)] =  -1.0;
        }

        void finalize() {
        }
    };


    /** connect global (scalar) variables to hoc -- */
    static DoubScal hoc_scalar_double[] = {
        {"c_thread_newton", &thread_newton_global.thread_data[0]},
        {nullptr, nullptr}
    };


    /** connect global (array) variables to hoc -- */
    static DoubVec hoc_vector_double[] = {
        {nullptr, nullptr, 0}
    };


    /* declaration of user functions */


    /* connect user functions to hoc names */
    static VoidFunc hoc_intfunc[] = {
        {"setdata_thread_newton", _hoc_setdata},
        {nullptr, nullptr}
    };
    static NPyDirectMechFunc npy_direct_func_proc[] = {
        {nullptr, nullptr}
    };
    static void thread_mem_init(Datum* _thread)  {
        if(thread_newton_global.thread_data_in_use) {
            _thread[0] = {neuron::container::do_not_search, new double[1]{}};
        }
        else {
            _thread[0] = {neuron::container::do_not_search, thread_newton_global.thread_data};
            thread_newton_global.thread_data_in_use = 1;
        }
    }
    static void thread_mem_cleanup(Datum* _thread)  {
        double * _thread_data_ptr = _thread[0].get<double*>();
        if(_thread_data_ptr == thread_newton_global.thread_data) {
            thread_newton_global.thread_data_in_use = 0;
        }
        else {
            delete[] _thread_data_ptr;
        }
    }


    void nrn_init_thread_newton(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _type};
        auto inst = make_instance_thread_newton(_lmc);
        auto node_data = make_node_data_thread_newton(*nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* _thread = _ml_arg->_thread;
        auto _thread_vars = thread_newton_ThreadVariables(_thread[0].get<double*>());
        for (int id = 0; id < nodecount; id++) {
            auto* _ppvar = _ml_arg->pdata[id];
            int node_id = node_data.nodeindices[id];
            auto v = node_data.node_voltages[node_id];
            inst.v_unused[id] = v;
            double total;
            inst.X[id] = 0.0;
            _thread_vars.c(id) = 42.0;
        }
    }


    void nrn_state_thread_newton(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _type};
        auto inst = make_instance_thread_newton(_lmc);
        auto node_data = make_node_data_thread_newton(*nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* _thread = _ml_arg->_thread;
        auto _thread_vars = thread_newton_ThreadVariables(_thread[0].get<double*>());
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_data.nodeindices[id];
            auto* _ppvar = _ml_arg->pdata[id];
            auto v = node_data.node_voltages[node_id];
            
            Eigen::Matrix<double, 1, 1> nmodl_eigen_xm;
            double* nmodl_eigen_x = nmodl_eigen_xm.data();
            nmodl_eigen_x[static_cast<int>(0)] = inst.X[id];
            // call newton solver
            functor_thread_newton_0 newton_functor(_lmc, inst, id, _ppvar, _thread, _thread_vars, nt, v);
            newton_functor.initialize();
            int newton_iterations = nmodl::newton::newton_solver(nmodl_eigen_xm, newton_functor);
            if (newton_iterations < 0) assert(false && "Newton solver did not converge!");
            inst.X[id] = nmodl_eigen_x[static_cast<int>(0)];
            newton_functor.finalize();

            inst.x[id] = inst.X[id];
        }
    }


    static void nrn_jacob_thread_newton(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _type};
        auto inst = make_instance_thread_newton(_lmc);
        auto node_data = make_node_data_thread_newton(*nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_data.nodeindices[id];
            node_data.node_diagonal[node_id] += inst.g_unused[id];
        }
    }


    static void _initlists() {
        /* X */
        _slist1[0] = {1, 0};
        /* DX */
        _dlist1[0] = {2, 0};
    }


    /** register channel with the simulator */
    extern "C" void _thread_newton_reg() {
        _initlists();

        register_mech(mechanism_info, nrn_alloc_thread_newton, nullptr, nrn_jacob_thread_newton, nrn_state_thread_newton, nrn_init_thread_newton, hoc_nrnpointerindex, 2);

        mech_type = nrn_get_mechtype(mechanism_info[1]);
        _nrn_mechanism_register_data_fields(mech_type,
            _nrn_mechanism_field<double>{"x"} /* 0 */,
            _nrn_mechanism_field<double>{"X"} /* 1 */,
            _nrn_mechanism_field<double>{"DX"} /* 2 */,
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
