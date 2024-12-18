/*********************************************************
Model Name      : heat_eqn_function
Filename        : heat_eqn_function.mod
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

/**
 * \dir
 * \brief Solver for a system of linear equations : Crout matrix decomposition
 *
 * \file
 * \brief Implementation of Crout matrix decomposition (LU decomposition) followed by
 * Forward/Backward substitution: Implementation details : (Legacy code) nrn / scopmath / crout.c
 */

#include <Eigen/Core>
#include <cmath>

#if defined(CORENEURON_ENABLE_GPU) && !defined(DISABLE_OPENACC)
#include "coreneuron/utils/offload.hpp"
#endif

namespace nmodl {
namespace crout {

/**
 * \brief Crout matrix decomposition : in-place LU Decomposition of matrix a.
 *
 * Implementation details : (Legacy code) nrn / scopmath / crout.c
 *
 * Returns: 0 if no error; -1 if matrix is singular or ill-conditioned
 */
#if defined(CORENEURON_ENABLE_GPU) && !defined(DISABLE_OPENACC)
nrn_pragma_acc(routine seq)
nrn_pragma_omp(declare target)
#endif
template <typename T>
EIGEN_DEVICE_FUNC inline int Crout(int n, T* const a, int* const perm, double* const rowmax) {
    // roundoff is the minimal value for a pivot element without its being considered too close to
    // zero
    double roundoff = 1.e-20;
    int i, j, k, r, pivot, irow, save_i = 0, krow;
    T sum, equil_1, equil_2;

    /* Initialize permutation and rowmax vectors */

    for (i = 0; i < n; i++) {
        perm[i] = i;
        k = 0;
        for (j = 1; j < n; j++) {
            if (std::fabs(a[i * n + j]) > std::fabs(a[i * n + k])) {
                k = j;
            }
        }
        rowmax[i] = a[i * n + k];
    }

    /* Loop over rows and columns r */

    for (r = 0; r < n; r++) {
        /*
         * Operate on rth column.  This produces the lower triangular matrix
         * of terms needed to transform the constant vector.
         */

        for (i = r; i < n; i++) {
            sum = 0.0;
            irow = perm[i];
            for (k = 0; k < r; k++) {
                krow = perm[k];
                sum += a[irow * n + k] * a[krow * n + r];
            }
            a[irow * n + r] -= sum;
        }

        /* Find row containing the pivot in the rth column */

        pivot = perm[r];
        equil_1 = std::fabs(a[pivot * n + r] / rowmax[pivot]);
        for (i = r + 1; i < n; i++) {
            irow = perm[i];
            equil_2 = std::fabs(a[irow * n + r] / rowmax[irow]);
            if (equil_2 > equil_1) {
                /* make irow the new pivot row */

                pivot = irow;
                save_i = i;
                equil_1 = equil_2;
            }
        }

        /* Interchange entries in permutation vector if necessary */

        if (pivot != perm[r]) {
            perm[save_i] = perm[r];
            perm[r] = pivot;
        }

        /* Check that pivot element is not too small */

        if (std::fabs(a[pivot * n + r]) < roundoff) {
            return -1;
        }

        /*
         * Operate on row in rth position.  This produces the upper
         * triangular matrix whose diagonal elements are assumed to be unity.
         * This matrix is used in the back substitution algorithm.
         */

        for (j = r + 1; j < n; j++) {
            sum = 0.0;
            for (k = 0; k < r; k++) {
                krow = perm[k];
                sum += a[pivot * n + k] * a[krow * n + j];
            }
            a[pivot * n + j] = (a[pivot * n + j] - sum) / a[pivot * n + r];
        }
    }
    return 0;
}
#if defined(CORENEURON_ENABLE_GPU) && !defined(DISABLE_OPENACC)
nrn_pragma_omp(end declare target)
#endif

/**
 * \brief Crout matrix decomposition : Forward/Backward substitution.
 *
 * Implementation details : (Legacy code) nrn / scopmath / crout.c
 *
 * Returns: no return variable
 */
#define y_(arg) p[y[arg]]
#define b_(arg) b[arg]
#if defined(CORENEURON_ENABLE_GPU) && !defined(DISABLE_OPENACC)
nrn_pragma_acc(routine seq)
nrn_pragma_omp(declare target)
#endif
template <typename T>
EIGEN_DEVICE_FUNC inline int solveCrout(int n,
                                        T const* const a,
                                        T const* const b,
                                        T* const p,
                                        int const* const perm,
                                        int const* const y = nullptr) {
    int i, j, pivot;
    T sum;

    /* Perform forward substitution with pivoting */
    if (y) {
        for (i = 0; i < n; i++) {
            pivot = perm[i];
            sum = 0.0;
            for (j = 0; j < i; j++) {
                sum += a[pivot * n + j] * (y_(j));
            }
            y_(i) = (b_(pivot) - sum) / a[pivot * n + i];
        }

        /*
         * Note that the y vector is already in the correct order for back
         * substitution.  Perform back substitution, pivoting the matrix but not
         * the y vector.  There is no need to divide by the diagonal element as
         * this is assumed to be unity.
         */

        for (i = n - 1; i >= 0; i--) {
            pivot = perm[i];
            sum = 0.0;
            for (j = i + 1; j < n; j++) {
                sum += a[pivot * n + j] * (y_(j));
            }
            y_(i) -= sum;
        }
    } else {
        for (i = 0; i < n; i++) {
            pivot = perm[i];
            sum = 0.0;
            for (j = 0; j < i; j++) {
                sum += a[pivot * n + j] * (p[j]);
            }
            p[i] = (b_(pivot) - sum) / a[pivot * n + i];
        }

        /*
         * Note that the y vector is already in the correct order for back
         * substitution.  Perform back substitution, pivoting the matrix but not
         * the y vector.  There is no need to divide by the diagonal element as
         * this is assumed to be unity.
         */

        for (i = n - 1; i >= 0; i--) {
            pivot = perm[i];
            sum = 0.0;
            for (j = i + 1; j < n; j++) {
                sum += a[pivot * n + j] * (p[j]);
            }
            p[i] -= sum;
        }
    }
    return 0;
}
#if defined(CORENEURON_ENABLE_GPU) && !defined(DISABLE_OPENACC)
nrn_pragma_omp(end declare target)
#endif

#undef y_
#undef b_

}  // namespace crout
}  // namespace nmodl

/**
 * \dir
 * \brief Newton solver implementations
 *
 * \file
 * \brief Implementation of Newton method for solving system of non-linear equations
 */

#include <Eigen/Dense>
#include <Eigen/LU>

namespace nmodl {
/// newton solver implementations
namespace newton {

/**
 * @defgroup solver Solver Implementation
 * @brief Solver implementation details
 *
 * Implementation of Newton method for solving system of non-linear equations using Eigen
 *   - newton::newton_solver with user, e.g. SymPy, provided Jacobian
 *
 * @{
 */

static constexpr int MAX_ITER = 50;
static constexpr double EPS = 1e-13;

template <int N>
EIGEN_DEVICE_FUNC bool is_converged(const Eigen::Matrix<double, N, 1>& X,
                                    const Eigen::Matrix<double, N, N>& J,
                                    const Eigen::Matrix<double, N, 1>& F,
                                    double eps) {
    bool converged = true;
    double square_eps = eps * eps;
    for (Eigen::Index i = 0; i < N; ++i) {
        double square_error = 0.0;
        for (Eigen::Index j = 0; j < N; ++j) {
            double JX = J(i, j) * X(j);
            square_error += JX * JX;
        }

        if (F(i) * F(i) > square_eps * square_error) {
            converged = false;
// The NVHPC is buggy and wont allow us to short-circuit.
#ifndef __NVCOMPILER
            return converged;
#endif
        }
    }
    return converged;
}

/**
 * \brief Newton method with user-provided Jacobian
 *
 * Newton method with user-provided Jacobian: given initial vector X and a
 * functor that calculates `F(X)`, `J(X)` where `J(X)` is the Jacobian of `F(X)`,
 * solves for \f$F(X) = 0\f$, starting with initial value of `X` by iterating:
 *
 *  \f[
 *     X_{n+1} = X_n - J(X_n)^{-1} F(X_n)
 *  \f]
 * when \f$|F|^2 < eps^2\f$, solution has converged.
 *
 * @return number of iterations (-1 if failed to converge)
 */
template <int N, typename FUNC>
EIGEN_DEVICE_FUNC int newton_solver(Eigen::Matrix<double, N, 1>& X,
                                    FUNC functor,
                                    double eps = EPS,
                                    int max_iter = MAX_ITER) {
    // If finite differences are needed, this is stores the stepwidth.
    Eigen::Matrix<double, N, 1> dX;
    // Vector to store result of function F(X):
    Eigen::Matrix<double, N, 1> F;
    // Matrix to store Jacobian of F(X):
    Eigen::Matrix<double, N, N> J;
    // Solver iteration count:
    int iter = -1;
    while (++iter < max_iter) {
        // calculate F, J from X using user-supplied functor
        functor(X, dX, F, J);
        if (is_converged(X, J, F, eps)) {
            return iter;
        }
        // In Eigen the default storage order is ColMajor.
        // Crout's implementation requires matrices stored in RowMajor order (C-style arrays).
        // Therefore, the transposeInPlace is critical such that the data() method to give the rows
        // instead of the columns.
        if (!J.IsRowMajor) {
            J.transposeInPlace();
        }
        Eigen::Matrix<int, N, 1> pivot;
        Eigen::Matrix<double, N, 1> rowmax;
        // Check if J is singular
        if (nmodl::crout::Crout<double>(N, J.data(), pivot.data(), rowmax.data()) < 0) {
            return -1;
        }
        Eigen::Matrix<double, N, 1> X_solve;
        nmodl::crout::solveCrout<double>(N, J.data(), F.data(), X_solve.data(), pivot.data());
        X -= X_solve;
    }
    // If we fail to converge after max_iter iterations, return -1
    return -1;
}

/**
 * Newton method template specializations for \f$N <= 4\f$ Use explicit inverse
 * of `F` instead of LU decomposition. This is faster, as there is no pivoting
 * and therefore no branches, but it is not numerically safe for \f$N > 4\f$.
 */

template <typename FUNC, int N>
EIGEN_DEVICE_FUNC int newton_solver_small_N(Eigen::Matrix<double, N, 1>& X,
                                            FUNC functor,
                                            double eps,
                                            int max_iter) {
    bool invertible;
    Eigen::Matrix<double, N, 1> F;
    Eigen::Matrix<double, N, 1> dX;
    Eigen::Matrix<double, N, N> J, J_inv;
    int iter = -1;
    while (++iter < max_iter) {
        functor(X, dX, F, J);
        if (is_converged(X, J, F, eps)) {
            return iter;
        }
        // The inverse can be called from within OpenACC regions without any issue, as opposed to
        // Eigen::PartialPivLU.
        J.computeInverseWithCheck(J_inv, invertible);
        if (invertible) {
            X -= J_inv * F;
        } else {
            return -1;
        }
    }
    return -1;
}

template <typename FUNC>
EIGEN_DEVICE_FUNC int newton_solver(Eigen::Matrix<double, 1, 1>& X,
                                    FUNC functor,
                                    double eps = EPS,
                                    int max_iter = MAX_ITER) {
    return newton_solver_small_N<FUNC, 1>(X, functor, eps, max_iter);
}

template <typename FUNC>
EIGEN_DEVICE_FUNC int newton_solver(Eigen::Matrix<double, 2, 1>& X,
                                    FUNC functor,
                                    double eps = EPS,
                                    int max_iter = MAX_ITER) {
    return newton_solver_small_N<FUNC, 2>(X, functor, eps, max_iter);
}

template <typename FUNC>
EIGEN_DEVICE_FUNC int newton_solver(Eigen::Matrix<double, 3, 1>& X,
                                    FUNC functor,
                                    double eps = EPS,
                                    int max_iter = MAX_ITER) {
    return newton_solver_small_N<FUNC, 3>(X, functor, eps, max_iter);
}

template <typename FUNC>
EIGEN_DEVICE_FUNC int newton_solver(Eigen::Matrix<double, 4, 1>& X,
                                    FUNC functor,
                                    double eps = EPS,
                                    int max_iter = MAX_ITER) {
    return newton_solver_small_N<FUNC, 4>(X, functor, eps, max_iter);
}

/** @} */  // end of solver

}  // namespace newton
}  // namespace nmodl


#include "mech_api.h"
#include "neuron/cache/mechanism_range.hpp"
#include "nmodlmutex.h"
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
extern void _cvode_abstol(Symbol**, double*, int);
extern Node* nrn_alloc_node_;


namespace neuron {
    #ifndef NRN_PRCELLSTATE
    #define NRN_PRCELLSTATE 0
    #endif


    /** channel information */
    static const char *mechanism_info[] = {
        "7.7.0",
        "heat_eqn_function",
        0,
        "x_heat_eqn_function",
        0,
        "X_heat_eqn_function",
        0,
        0
    };


    /* NEURON global variables */
    static neuron::container::field_index _slist1[1], _dlist1[1];
    static int mech_type;
    static Prop* _extcall_prop;
    /* _prop_id kind of shadows _extcall_prop to allow validity checking. */
    static _nrn_non_owning_id_without_container _prop_id{};
    static _nrn_mechanism_std_vector<Datum> _extcall_thread;


    /** all global variables */
    struct heat_eqn_function_Store {
        double thread_data_in_use{0};
        double thread_data[2] /* TODO init const-array */;
        double X0{0};
    };
    static_assert(std::is_trivially_copy_constructible_v<heat_eqn_function_Store>);
    static_assert(std::is_trivially_move_constructible_v<heat_eqn_function_Store>);
    static_assert(std::is_trivially_copy_assignable_v<heat_eqn_function_Store>);
    static_assert(std::is_trivially_move_assignable_v<heat_eqn_function_Store>);
    static_assert(std::is_trivially_destructible_v<heat_eqn_function_Store>);
    static heat_eqn_function_Store heat_eqn_function_global;
    auto thread_data_in_use_heat_eqn_function() -> std::decay<decltype(heat_eqn_function_global.thread_data_in_use)>::type  {
        return heat_eqn_function_global.thread_data_in_use;
    }
    auto thread_data_heat_eqn_function() -> std::decay<decltype(heat_eqn_function_global.thread_data)>::type  {
        return heat_eqn_function_global.thread_data;
    }
    auto X0_heat_eqn_function() -> std::decay<decltype(heat_eqn_function_global.X0)>::type  {
        return heat_eqn_function_global.X0;
    }

    static std::vector<double> _parameter_defaults = {
    };


    /** all mechanism instance variables and global variables */
    struct heat_eqn_function_Instance  {
        double* x{};
        double* X{};
        double* DX{};
        double* v_unused{};
        double* g_unused{};
        heat_eqn_function_Store* global{&heat_eqn_function_global};
    };


    struct heat_eqn_function_NodeData  {
        int const * nodeindices;
        double const * node_voltages;
        double * node_diagonal;
        double * node_rhs;
        int nodecount;
    };


    struct heat_eqn_function_ThreadVariables  {
        double * thread_data;

        double * g_vol_ptr(size_t id) {
            return thread_data + 0 + (id % 1);
        }
        double & g_vol(size_t id) {
            return thread_data[0 + (id % 1)];
        }
        double * g_mu_ptr(size_t id) {
            return thread_data + 1 + (id % 1);
        }
        double & g_mu(size_t id) {
            return thread_data[1 + (id % 1)];
        }

        heat_eqn_function_ThreadVariables(double * const thread_data) {
            this->thread_data = thread_data;
        }
    };


    static heat_eqn_function_Instance make_instance_heat_eqn_function(_nrn_mechanism_cache_range* _lmc) {
        if(_lmc == nullptr) {
            return heat_eqn_function_Instance();
        }

        return heat_eqn_function_Instance {
            _lmc->template fpfield_ptr<0>(),
            _lmc->template fpfield_ptr<1>(),
            _lmc->template fpfield_ptr<2>(),
            _lmc->template fpfield_ptr<3>(),
            _lmc->template fpfield_ptr<4>()
        };
    }


    static heat_eqn_function_NodeData make_node_data_heat_eqn_function(NrnThread& nt, Memb_list& _ml_arg) {
        return heat_eqn_function_NodeData {
            _ml_arg.nodeindices,
            nt.node_voltage_storage(),
            nt.node_d_storage(),
            nt.node_rhs_storage(),
            _ml_arg.nodecount
        };
    }
    static heat_eqn_function_NodeData make_node_data_heat_eqn_function(Prop * _prop) {
        if(!_prop) {
            return heat_eqn_function_NodeData();
        }

        static std::vector<int> node_index{0};
        Node* _node = _nrn_mechanism_access_node(_prop);
        return heat_eqn_function_NodeData {
            node_index.data(),
            &_nrn_mechanism_access_voltage(_node),
            &_nrn_mechanism_access_d(_node),
            &_nrn_mechanism_access_rhs(_node),
            1
        };
    }

    static void nrn_destructor_heat_eqn_function(Prop* prop);


    static void nrn_alloc_heat_eqn_function(Prop* _prop) {
        Datum *_ppvar = nullptr;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const _iml = 0;
        assert(_nrn_mechanism_get_num_vars(_prop) == 5);
        /*initialize range parameters*/
    }


    /* Mechanism procedures and functions */
    inline static double factor_heat_eqn_function(_nrn_mechanism_cache_range& _lmc, heat_eqn_function_Instance& inst, heat_eqn_function_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, heat_eqn_function_ThreadVariables& _thread_vars, NrnThread* nt, double _lx);
    inline static double vol_heat_eqn_function(_nrn_mechanism_cache_range& _lmc, heat_eqn_function_Instance& inst, heat_eqn_function_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, heat_eqn_function_ThreadVariables& _thread_vars, NrnThread* nt, double _lx);
    inline static double mu_heat_eqn_function(_nrn_mechanism_cache_range& _lmc, heat_eqn_function_Instance& inst, heat_eqn_function_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, heat_eqn_function_ThreadVariables& _thread_vars, NrnThread* nt, double _lx);
    static void* _diffusion_space_X;
    static double _diffusion_coefficient_X(int _i, Memb_list* _ml_arg, size_t id, Datum* _ppvar, double* _pdvol, double* _pdfcdc, Datum* /* _thread */, NrnThread* nt, const _nrn_model_sorted_token& _sorted_token) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_heat_eqn_function(&_lmc);
        auto node_data = make_node_data_heat_eqn_function(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto _thread_vars = heat_eqn_function_ThreadVariables(_thread[0].get<double*>());
        *_pdvol= vol_heat_eqn_function(_lmc, inst, node_data, id, _ppvar, _thread, _thread_vars, nt, inst.x[id]);
        *_pdfcdc = 0.0;
        return mu_heat_eqn_function(_lmc, inst, node_data, id, _ppvar, _thread, _thread_vars, nt, factor_heat_eqn_function(_lmc, inst, node_data, id, _ppvar, _thread, _thread_vars, nt, inst.x[id]));
    }
    static void _apply_diffusion_function(ldifusfunc2_t _f, const _nrn_model_sorted_token& _sorted_token, NrnThread& _nt) {
        for(size_t _i = 0; _i < 1; ++_i) {
            (*_f)(mech_type, _diffusion_coefficient_X, &_diffusion_space_X, _i, /* x pos */ 1, /* Dx pos */ 2, _sorted_token, _nt);
        }
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


    struct functor_heat_eqn_function_0 {
        _nrn_mechanism_cache_range& _lmc;
        heat_eqn_function_Instance& inst;
        heat_eqn_function_NodeData& node_data;
        size_t id;
        Datum* _ppvar;
        Datum* _thread;
        heat_eqn_function_ThreadVariables& _thread_vars;
        NrnThread* nt;
        double source0_, old_X;

        void initialize() {
            ;
            source0_ = 0.0;
            old_X = inst.X[id];
        }

        functor_heat_eqn_function_0(_nrn_mechanism_cache_range& _lmc, heat_eqn_function_Instance& inst, heat_eqn_function_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, heat_eqn_function_ThreadVariables& _thread_vars, NrnThread* nt)
            : _lmc(_lmc), inst(inst), node_data(node_data), id(id), _ppvar(_ppvar), _thread(_thread), _thread_vars(_thread_vars), nt(nt)
        {}
        void operator()(const Eigen::Matrix<double, 1, 1>& nmodl_eigen_xm, Eigen::Matrix<double, 1, 1>& nmodl_eigen_dxm, Eigen::Matrix<double, 1, 1>& nmodl_eigen_fm, Eigen::Matrix<double, 1, 1>& nmodl_eigen_jm) const {
            const double* nmodl_eigen_x = nmodl_eigen_xm.data();
            double* nmodl_eigen_dx = nmodl_eigen_dxm.data();
            double* nmodl_eigen_j = nmodl_eigen_jm.data();
            double* nmodl_eigen_f = nmodl_eigen_fm.data();
            nmodl_eigen_dx[0] = std::max(1e-6, 0.02*std::fabs(nmodl_eigen_x[0]));
            nmodl_eigen_f[static_cast<int>(0)] =  -nmodl_eigen_x[static_cast<int>(0)] / nt->_dt + source0_ / vol_heat_eqn_function(_lmc, inst, node_data, id, _ppvar, _thread, _thread_vars, nt, inst.x[id]) + old_X / nt->_dt;
            nmodl_eigen_j[static_cast<int>(0)] =  -1.0 / nt->_dt;
        }

        void finalize() {
        }
    };


    /** connect global (scalar) variables to hoc -- */
    static DoubScal hoc_scalar_double[] = {
        {"g_vol_heat_eqn_function", &heat_eqn_function_global.thread_data[0]},
        {"g_mu_heat_eqn_function", &heat_eqn_function_global.thread_data[1]},
        {nullptr, nullptr}
    };


    /** connect global (array) variables to hoc -- */
    static DoubVec hoc_vector_double[] = {
        {nullptr, nullptr, 0}
    };


    /* declaration of user functions */
    static void _hoc_factor();
    static double _npy_factor(Prop* _prop);
    static void _hoc_vol();
    static double _npy_vol(Prop* _prop);
    static void _hoc_mu();
    static double _npy_mu(Prop* _prop);


    /* connect user functions to hoc names */
    static VoidFunc hoc_intfunc[] = {
        {"setdata_heat_eqn_function", _hoc_setdata},
        {"factor_heat_eqn_function", _hoc_factor},
        {"vol_heat_eqn_function", _hoc_vol},
        {"mu_heat_eqn_function", _hoc_mu},
        {nullptr, nullptr}
    };
    static NPyDirectMechFunc npy_direct_func_proc[] = {
        {"factor", _npy_factor},
        {"vol", _npy_vol},
        {"mu", _npy_mu},
        {nullptr, nullptr}
    };
    static void thread_mem_init(Datum* _thread)  {
        if(heat_eqn_function_global.thread_data_in_use) {
            _thread[0] = {neuron::container::do_not_search, new double[2]{}};
        }
        else {
            _thread[0] = {neuron::container::do_not_search, heat_eqn_function_global.thread_data};
            heat_eqn_function_global.thread_data_in_use = 1;
        }
    }
    static void thread_mem_cleanup(Datum* _thread)  {
        double * _thread_data_ptr = _thread[0].get<double*>();
        if(_thread_data_ptr == heat_eqn_function_global.thread_data) {
            heat_eqn_function_global.thread_data_in_use = 0;
        }
        else {
            delete[] _thread_data_ptr;
        }
    }
    static void _hoc_factor() {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        Prop* _local_prop = _prop_id ? _extcall_prop : nullptr;
        _nrn_mechanism_cache_instance _lmc{_local_prop};
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_heat_eqn_function(_local_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_heat_eqn_function(_local_prop);
        auto _thread_vars = heat_eqn_function_ThreadVariables(_thread[0].get<double*>());
        double _r = 0.0;
        _r = factor_heat_eqn_function(_lmc, inst, node_data, id, _ppvar, _thread, _thread_vars, nt, *getarg(1));
        hoc_retpushx(_r);
    }
    static double _npy_factor(Prop* _prop) {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id = 0;
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_heat_eqn_function(_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_heat_eqn_function(_prop);
        auto _thread_vars = heat_eqn_function_ThreadVariables(_thread[0].get<double*>());
        double _r = 0.0;
        _r = factor_heat_eqn_function(_lmc, inst, node_data, id, _ppvar, _thread, _thread_vars, nt, *getarg(1));
        return(_r);
    }
    static void _hoc_vol() {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        Prop* _local_prop = _prop_id ? _extcall_prop : nullptr;
        _nrn_mechanism_cache_instance _lmc{_local_prop};
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_heat_eqn_function(_local_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_heat_eqn_function(_local_prop);
        auto _thread_vars = heat_eqn_function_ThreadVariables(_thread[0].get<double*>());
        double _r = 0.0;
        _r = vol_heat_eqn_function(_lmc, inst, node_data, id, _ppvar, _thread, _thread_vars, nt, *getarg(1));
        hoc_retpushx(_r);
    }
    static double _npy_vol(Prop* _prop) {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id = 0;
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_heat_eqn_function(_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_heat_eqn_function(_prop);
        auto _thread_vars = heat_eqn_function_ThreadVariables(_thread[0].get<double*>());
        double _r = 0.0;
        _r = vol_heat_eqn_function(_lmc, inst, node_data, id, _ppvar, _thread, _thread_vars, nt, *getarg(1));
        return(_r);
    }
    static void _hoc_mu() {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        Prop* _local_prop = _prop_id ? _extcall_prop : nullptr;
        _nrn_mechanism_cache_instance _lmc{_local_prop};
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_heat_eqn_function(_local_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_heat_eqn_function(_local_prop);
        auto _thread_vars = heat_eqn_function_ThreadVariables(_thread[0].get<double*>());
        double _r = 0.0;
        _r = mu_heat_eqn_function(_lmc, inst, node_data, id, _ppvar, _thread, _thread_vars, nt, *getarg(1));
        hoc_retpushx(_r);
    }
    static double _npy_mu(Prop* _prop) {
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id = 0;
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_heat_eqn_function(_prop ? &_lmc : nullptr);
        auto node_data = make_node_data_heat_eqn_function(_prop);
        auto _thread_vars = heat_eqn_function_ThreadVariables(_thread[0].get<double*>());
        double _r = 0.0;
        _r = mu_heat_eqn_function(_lmc, inst, node_data, id, _ppvar, _thread, _thread_vars, nt, *getarg(1));
        return(_r);
    }


    inline double factor_heat_eqn_function(_nrn_mechanism_cache_range& _lmc, heat_eqn_function_Instance& inst, heat_eqn_function_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, heat_eqn_function_ThreadVariables& _thread_vars, NrnThread* nt, double _lx) {
        double ret_factor = 0.0;
        double v = node_data.node_voltages ? node_data.node_voltages[node_data.nodeindices[id]] : 0.0;
        if (_lx < 0.25) {
            ret_factor = 0.0;
        } else {
            ret_factor = 10.0 * (_lx - 0.25);
        }
        return ret_factor;
    }


    inline double vol_heat_eqn_function(_nrn_mechanism_cache_range& _lmc, heat_eqn_function_Instance& inst, heat_eqn_function_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, heat_eqn_function_ThreadVariables& _thread_vars, NrnThread* nt, double _lx) {
        double ret_vol = 0.0;
        double v = node_data.node_voltages ? node_data.node_voltages[node_data.nodeindices[id]] : 0.0;
        ret_vol = (1.0 + _lx) * _thread_vars.g_vol(id);
        return ret_vol;
    }


    inline double mu_heat_eqn_function(_nrn_mechanism_cache_range& _lmc, heat_eqn_function_Instance& inst, heat_eqn_function_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, heat_eqn_function_ThreadVariables& _thread_vars, NrnThread* nt, double _lx) {
        double ret_mu = 0.0;
        double v = node_data.node_voltages ? node_data.node_voltages[node_data.nodeindices[id]] : 0.0;
        ret_mu = _lx * _thread_vars.g_mu(id);
        return ret_mu;
    }


    static void nrn_init_heat_eqn_function(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_heat_eqn_function(&_lmc);
        auto node_data = make_node_data_heat_eqn_function(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto _thread_vars = heat_eqn_function_ThreadVariables(_thread[0].get<double*>());
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
            auto* _ppvar = _ml_arg->pdata[id];
            int node_id = node_data.nodeindices[id];
            inst.v_unused[id] = node_data.node_voltages[node_id];
            inst.X[id] = inst.global->X0;
            _thread_vars.g_mu(id) = 1.1;
            _thread_vars.g_vol(id) = 0.01;
            if (inst.x[id] < 0.5) {
                inst.X[id] = 1.0;
            } else {
                inst.X[id] = 0.0;
            }
        }
    }


    static void nrn_state_heat_eqn_function(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_heat_eqn_function(&_lmc);
        auto node_data = make_node_data_heat_eqn_function(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto _thread_vars = heat_eqn_function_ThreadVariables(_thread[0].get<double*>());
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_data.nodeindices[id];
            auto* _ppvar = _ml_arg->pdata[id];
            inst.v_unused[id] = node_data.node_voltages[node_id];
            
            Eigen::Matrix<double, 1, 1> nmodl_eigen_xm;
            double* nmodl_eigen_x = nmodl_eigen_xm.data();
            nmodl_eigen_x[static_cast<int>(0)] = inst.X[id];
            // call newton solver
            functor_heat_eqn_function_0 newton_functor(_lmc, inst, node_data, id, _ppvar, _thread, _thread_vars, nt);
            newton_functor.initialize();
            int newton_iterations = nmodl::newton::newton_solver(nmodl_eigen_xm, newton_functor);
            if (newton_iterations < 0) assert(false && "Newton solver did not converge!");
            inst.X[id] = nmodl_eigen_x[static_cast<int>(0)];
            newton_functor.initialize(); // TODO mimic calling F again.
            newton_functor.finalize();

        }
    }


    static void nrn_jacob_heat_eqn_function(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_heat_eqn_function(&_lmc);
        auto node_data = make_node_data_heat_eqn_function(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto _thread_vars = heat_eqn_function_ThreadVariables(_thread[0].get<double*>());
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_data.nodeindices[id];
            node_data.node_diagonal[node_id] += inst.g_unused[id];
        }
    }
    static void nrn_destructor_heat_eqn_function(Prop* prop) {
        Datum* _ppvar = _nrn_mechanism_access_dparam(prop);
        _nrn_mechanism_cache_instance _lmc{prop};
        const size_t id = 0;
        auto inst = make_instance_heat_eqn_function(prop ? &_lmc : nullptr);
        auto node_data = make_node_data_heat_eqn_function(prop);
        auto _thread_vars = heat_eqn_function_ThreadVariables(heat_eqn_function_global.thread_data);

    }


    static void _initlists() {
        /* X */
        _slist1[0] = {1, 0};
        /* DX */
        _dlist1[0] = {2, 0};
    }


    extern "C" void _heat_eqn_function_reg() {
        _initlists();

        register_mech(mechanism_info, nrn_alloc_heat_eqn_function, nullptr, nrn_jacob_heat_eqn_function, nrn_state_heat_eqn_function, nrn_init_heat_eqn_function, -1, 2);
        _extcall_thread.resize(2);
        thread_mem_init(_extcall_thread.data());
        heat_eqn_function_global.thread_data_in_use = 0;

        mech_type = nrn_get_mechtype(mechanism_info[1]);
        hoc_register_parm_default(mech_type, &_parameter_defaults);
        _nrn_mechanism_register_data_fields(mech_type,
            _nrn_mechanism_field<double>{"x"} /* 0 */,
            _nrn_mechanism_field<double>{"X"} /* 1 */,
            _nrn_mechanism_field<double>{"DX"} /* 2 */,
            _nrn_mechanism_field<double>{"v_unused"} /* 3 */,
            _nrn_mechanism_field<double>{"g_unused"} /* 4 */
        );

        hoc_register_prop_size(mech_type, 5, 0);
        hoc_register_ldifus1(_apply_diffusion_function);
        hoc_register_var(hoc_scalar_double, hoc_vector_double, hoc_intfunc);
        hoc_register_npy_direct(mech_type, npy_direct_func_proc);
        _nrn_thread_reg(mech_type, 1, thread_mem_init);
        _nrn_thread_reg(mech_type, 0, thread_mem_cleanup);
    }
}
