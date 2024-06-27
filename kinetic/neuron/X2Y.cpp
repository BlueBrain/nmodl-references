/*********************************************************
Model Name      : X2Y
Filename        : X2Y.mod
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
#include <crout/crout.hpp>
#include <math.h>
#include <newton/newton.hpp>
#include <stdio.h>
#include <stdlib.h>

#include "mech_api.h"
#include "neuron/cache/mechanism_range.hpp"
#include "nrniv_mf.h"
#include "section_fwd.hpp"

/* NEURON global macro definitions */
/* VECTORIZED */
#define NRN_VECTORIZED 1

static constexpr auto number_of_datum_variables = 0;
static constexpr auto number_of_floating_point_variables = 8;

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
        "X2Y",
        0,
        "il_X2Y",
        0,
        "X_X2Y",
        "Y_X2Y",
        0,
        0
    };


    /* NEURON global variables */
    static neuron::container::field_index _slist1[2], _dlist1[2];
    static int mech_type;
    static Prop* _extcall_prop;
    /* _prop_id kind of shadows _extcall_prop to allow validity checking. */
    static _nrn_non_owning_id_without_container _prop_id{};
    static int hoc_nrnpointerindex = -1;
    static _nrn_mechanism_std_vector<Datum> _extcall_thread;


    /** all global variables */
    struct X2Y_Store {
        double X0{};
        double Y0{};
    };
    static_assert(std::is_trivially_copy_constructible_v<X2Y_Store>);
    static_assert(std::is_trivially_move_constructible_v<X2Y_Store>);
    static_assert(std::is_trivially_copy_assignable_v<X2Y_Store>);
    static_assert(std::is_trivially_move_assignable_v<X2Y_Store>);
    static_assert(std::is_trivially_destructible_v<X2Y_Store>);
    X2Y_Store X2Y_global;


    /** all mechanism instance variables and global variables */
    struct X2Y_Instance  {
        double* il{};
        double* X{};
        double* Y{};
        double* DX{};
        double* DY{};
        double* i{};
        double* v_unused{};
        double* g_unused{};
        X2Y_Store* global{&X2Y_global};
    };


    struct X2Y_NodeData  {
        int const * nodeindices;
        double const * node_voltages;
        double * node_diagonal;
        double * node_rhs;
        int nodecount;
    };


    static X2Y_Instance make_instance_X2Y(_nrn_mechanism_cache_range& _lmc) {
        return X2Y_Instance {
            _lmc.template fpfield_ptr<0>(),
            _lmc.template fpfield_ptr<1>(),
            _lmc.template fpfield_ptr<2>(),
            _lmc.template fpfield_ptr<3>(),
            _lmc.template fpfield_ptr<4>(),
            _lmc.template fpfield_ptr<5>(),
            _lmc.template fpfield_ptr<6>(),
            _lmc.template fpfield_ptr<7>()
        };
    }


    static X2Y_NodeData make_node_data_X2Y(NrnThread& nt, Memb_list& _ml_arg) {
        return X2Y_NodeData {
            _ml_arg.nodeindices,
            nt.node_voltage_storage(),
            nt.node_d_storage(),
            nt.node_rhs_storage(),
            _ml_arg.nodecount
        };
    }


    static void nrn_alloc_X2Y(Prop* _prop) {
        Prop *prop_ion{};
        Datum *_ppvar{};
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const _iml{};
        assert(_nrn_mechanism_get_num_vars(_prop) == 8);
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


    struct functor_X2Y_0 {
        NrnThread* nt;
        X2Y_Instance& inst;
        int id;
        double v;
        Datum* _thread;
        double kf0_, kb0_, old_X, old_Y;

        void initialize() {
            kf0_ = 0.4;
            kb0_ = 0.5;
            inst.i[id] = (kf0_ * inst.X[id] - kb0_ * inst.Y[id]);
            old_X = inst.X[id];
            old_Y = inst.Y[id];
        }

        functor_X2Y_0(NrnThread* nt, X2Y_Instance& inst, int id, double v, Datum* _thread)
            : nt(nt), inst(inst), id(id), v(v), _thread(_thread)
        {}
        void operator()(const Eigen::Matrix<double, 2, 1>& nmodl_eigen_xm, Eigen::Matrix<double, 2, 1>& nmodl_eigen_fm, Eigen::Matrix<double, 2, 2>& nmodl_eigen_jm) const {
            const double* nmodl_eigen_x = nmodl_eigen_xm.data();
            double* nmodl_eigen_j = nmodl_eigen_jm.data();
            double* nmodl_eigen_f = nmodl_eigen_fm.data();
            nmodl_eigen_f[static_cast<int>(0)] =  -nmodl_eigen_x[static_cast<int>(0)] * nt->_dt * kf0_ - nmodl_eigen_x[static_cast<int>(0)] + nmodl_eigen_x[static_cast<int>(1)] * nt->_dt * kb0_ + old_X;
            nmodl_eigen_j[static_cast<int>(0)] =  -nt->_dt * kf0_ - 1.0;
            nmodl_eigen_j[static_cast<int>(2)] = nt->_dt * kb0_;
            nmodl_eigen_f[static_cast<int>(1)] = nmodl_eigen_x[static_cast<int>(0)] * nt->_dt * kf0_ - nmodl_eigen_x[static_cast<int>(1)] * nt->_dt * kb0_ - nmodl_eigen_x[static_cast<int>(1)] + old_Y;
            nmodl_eigen_j[static_cast<int>(1)] = nt->_dt * kf0_;
            nmodl_eigen_j[static_cast<int>(3)] =  -nt->_dt * kb0_ - 1.0;
        }

        void finalize() {
        }
    };


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
        {"setdata_X2Y", _hoc_setdata},
        {nullptr, nullptr}
    };
    static NPyDirectMechFunc npy_direct_func_proc[] = {
        {nullptr, nullptr}
    };


    void nrn_init_X2Y(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _type};
        auto inst = make_instance_X2Y(_lmc);
        auto node_data = make_node_data_X2Y(*nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* _thread = _ml_arg->_thread;
        for (int id = 0; id < nodecount; id++) {
            auto* _ppvar = _ml_arg->pdata[id];
            int node_id = node_data.nodeindices[id];
            auto v = node_data.node_voltages[node_id];
            inst.v_unused[id] = v;
            inst.X[id] = 0.0;
            inst.Y[id] = 1.0;
        }
    }


    inline double nrn_current_X2Y(_nrn_mechanism_cache_range& _lmc, NrnThread* nt, Datum* _ppvar, Datum* _thread, size_t id, X2Y_Instance& inst, X2Y_NodeData& node_data, double v) {
        double current = 0.0;
        inst.il[id] = inst.i[id];
        current += inst.il[id];
        return current;
    }


    /** update current */
    void nrn_cur_X2Y(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _type};
        auto inst = make_instance_X2Y(_lmc);
        auto node_data = make_node_data_X2Y(*nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* _thread = _ml_arg->_thread;
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_data.nodeindices[id];
            double v = node_data.node_voltages[node_id];
            auto* _ppvar = _ml_arg->pdata[id];
            double I1 = nrn_current_X2Y(_lmc, nt, _ppvar, _thread, id, inst, node_data, v+0.001);
            double I0 = nrn_current_X2Y(_lmc, nt, _ppvar, _thread, id, inst, node_data, v);
            double rhs = I0;
            double g = (I1-I0)/0.001;
            node_data.node_rhs[node_id] -= rhs;
            inst.g_unused[id] = g;
        }
    }


    void nrn_state_X2Y(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _type};
        auto inst = make_instance_X2Y(_lmc);
        auto node_data = make_node_data_X2Y(*nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* _thread = _ml_arg->_thread;
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_data.nodeindices[id];
            auto* _ppvar = _ml_arg->pdata[id];
            auto v = node_data.node_voltages[node_id];
            
            Eigen::Matrix<double, 2, 1> nmodl_eigen_xm;
            double* nmodl_eigen_x = nmodl_eigen_xm.data();
            nmodl_eigen_x[static_cast<int>(0)] = inst.X[id];
            nmodl_eigen_x[static_cast<int>(1)] = inst.Y[id];
            // call newton solver
            functor_X2Y_0 newton_functor(nt, inst, id, v, _thread);
            newton_functor.initialize();
            int newton_iterations = nmodl::newton::newton_solver(nmodl_eigen_xm, newton_functor);
            if (newton_iterations < 0) assert(false && "Newton solver did not converge!");
            inst.X[id] = nmodl_eigen_x[static_cast<int>(0)];
            inst.Y[id] = nmodl_eigen_x[static_cast<int>(1)];
            newton_functor.finalize();

        }
    }


    static void nrn_jacob_X2Y(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _type};
        auto inst = make_instance_X2Y(_lmc);
        auto node_data = make_node_data_X2Y(*nt, *_ml_arg);
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
        _dlist1[0] = {3, 0};
        /* Y */
        _slist1[1] = {2, 0};
        /* DY */
        _dlist1[1] = {4, 0};
    }


    /** register channel with the simulator */
    extern "C" void _X2Y_reg() {
        _initlists();

        register_mech(mechanism_info, nrn_alloc_X2Y, nrn_cur_X2Y, nrn_jacob_X2Y, nrn_state_X2Y, nrn_init_X2Y, hoc_nrnpointerindex, 1);

        mech_type = nrn_get_mechtype(mechanism_info[1]);
        _nrn_mechanism_register_data_fields(mech_type,
            _nrn_mechanism_field<double>{"il"} /* 0 */,
            _nrn_mechanism_field<double>{"X"} /* 1 */,
            _nrn_mechanism_field<double>{"Y"} /* 2 */,
            _nrn_mechanism_field<double>{"DX"} /* 3 */,
            _nrn_mechanism_field<double>{"DY"} /* 4 */,
            _nrn_mechanism_field<double>{"i"} /* 5 */,
            _nrn_mechanism_field<double>{"v_unused"} /* 6 */,
            _nrn_mechanism_field<double>{"g_unused"} /* 7 */
        );

        hoc_register_prop_size(mech_type, 8, 0);
        hoc_register_var(hoc_scalar_double, hoc_vector_double, hoc_intfunc);
        hoc_register_npy_direct(mech_type, npy_direct_func_proc);
    }
}
