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

#include <Eigen/Dense>
#include <Eigen/LU>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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
        for (j = 1; j < n; j++)
            if (std::fabs(a[i * n + j]) > std::fabs(a[i * n + k]))
                k = j;
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

        if (std::fabs(a[pivot * n + r]) < roundoff)
            return -1;

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
            for (j = 0; j < i; j++)
                sum += a[pivot * n + j] * (y_(j));
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
            for (j = i + 1; j < n; j++)
                sum += a[pivot * n + j] * (y_(j));
            y_(i) -= sum;
        }
    } else {
        for (i = 0; i < n; i++) {
            pivot = perm[i];
            sum = 0.0;
            for (j = 0; j < i; j++)
                sum += a[pivot * n + j] * (p[j]);
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
            for (j = i + 1; j < n; j++)
                sum += a[pivot * n + j] * (p[j]);
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
 *   - newton::newton_solver is the preferred option: requires user to provide Jacobian
 *   - newton::newton_numerical_diff_solver is the fallback option: Jacobian not required
 *
 * @{
 */

static constexpr int MAX_ITER = 1e3;
static constexpr double EPS = 1e-12;

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
    // Vector to store result of function F(X):
    Eigen::Matrix<double, N, 1> F;
    // Matrix to store jacobian of F(X):
    Eigen::Matrix<double, N, N> J;
    // Solver iteration count:
    int iter = -1;
    while (++iter < max_iter) {
        // calculate F, J from X using user-supplied functor
        functor(X, F, J);
        // get error norm: here we use sqrt(|F|^2)
        double error = F.norm();
        if (error < eps) {
            // we have converged: return iteration count
            return iter;
        }
        // In Eigen the default storage order is ColMajor.
        // Crout's implementation requires matrices stored in RowMajor order (C-style arrays).
        // Therefore, the transposeInPlace is critical such that the data() method to give the rows
        // instead of the columns.
        if (!J.IsRowMajor)
            J.transposeInPlace();
        Eigen::Matrix<int, N, 1> pivot;
        Eigen::Matrix<double, N, 1> rowmax;
        // Check if J is singular
        if (nmodl::crout::Crout<double>(N, J.data(), pivot.data(), rowmax.data()) < 0)
            return -1;
        Eigen::Matrix<double, N, 1> X_solve;
        nmodl::crout::solveCrout<double>(N, J.data(), F.data(), X_solve.data(), pivot.data());
        X -= X_solve;
    }
    // If we fail to converge after max_iter iterations, return -1
    return -1;
}

static constexpr double SQUARE_ROOT_ULP = 1e-7;
static constexpr double CUBIC_ROOT_ULP = 1e-5;

/**
 * \brief Newton method without user-provided Jacobian
 *
 * Newton method without user-provided Jacobian: given initial vector X and a
 * functor that calculates `F(X)`, solves for \f$F(X) = 0\f$, starting with
 * initial value of `X` by iterating:
 *
 * \f[
 *     X_{n+1} = X_n - J(X_n)^{-1} F(X_n)
 * \f]
 *
 * where `J(X)` is the Jacobian of `F(X)`, which is approximated numerically
 * using a symmetric finite difference approximation to the derivative
 * when \f$|F|^2 < eps^2\f$, solution has converged/
 *
 * @return number of iterations (-1 if failed to converge)
 */
template <int N, typename FUNC>
EIGEN_DEVICE_FUNC int newton_numerical_diff_solver(Eigen::Matrix<double, N, 1>& X,
                                                   FUNC functor,
                                                   double eps = EPS,
                                                   int max_iter = MAX_ITER) {
    // Vector to store result of function F(X):
    Eigen::Matrix<double, N, 1> F;
    // Temporary storage for F(X+dx)
    Eigen::Matrix<double, N, 1> F_p;
    // Temporary storage for F(X-dx)
    Eigen::Matrix<double, N, 1> F_m;
    // Matrix to store jacobian of F(X):
    Eigen::Matrix<double, N, N> J;
    // Solver iteration count:
    int iter = 0;
    while (iter < max_iter) {
        // calculate F from X using user-supplied functor
        functor(X, F);
        // get error norm: here we use sqrt(|F|^2)
        double error = F.norm();
        if (error < eps) {
            // we have converged: return iteration count
            return iter;
        }
        ++iter;
        // calculate approximate Jacobian
        for (int i = 0; i < N; ++i) {
            // symmetric finite difference approximation to derivative
            // df/dx ~= ( f(x+dx) - f(x-dx) ) / (2*dx)
            // choose dx to be ~(ULP)^{1/3}*X[i]
            // https://aip.scitation.org/doi/pdf/10.1063/1.4822971
            // also enforce a lower bound ~sqrt(ULP) to avoid dx being too small
            double dX = std::max(CUBIC_ROOT_ULP * X[i], SQUARE_ROOT_ULP);
            // F(X + dX)
            X[i] += dX;
            functor(X, F_p);
            // F(X - dX)
            X[i] -= 2.0 * dX;
            functor(X, F_m);
            F_p -= F_m;
            // J = (F(X + dX) - F(X - dX)) / (2*dX)
            J.col(i) = F_p / (2.0 * dX);
            // restore X
            X[i] += dX;
        }
        if (!J.IsRowMajor)
            J.transposeInPlace();
        Eigen::Matrix<int, N, 1> pivot;
        Eigen::Matrix<double, N, 1> rowmax;
        // Check if J is singular
        if (nmodl::crout::Crout<double>(N, J.data(), pivot.data(), rowmax.data()) < 0)
            return -1;
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
    Eigen::Matrix<double, N, N> J, J_inv;
    int iter = -1;
    while (++iter < max_iter) {
        functor(X, F, J);
        double error = F.norm();
        if (error < eps) {
            return iter;
        }
        // The inverse can be called from within OpenACC regions without any issue, as opposed to
        // Eigen::PartialPivLU.
        J.computeInverseWithCheck(J_inv, invertible);
        if (invertible)
            X -= J_inv * F;
        else
            return -1;
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
void _nrn_thread_table_reg(int, nrn_thread_table_check_t);
extern Node* nrn_alloc_node_;


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
        double thread_data[5];
        double usetable{1};
        double tmin_compute_g_v1{};
        double mfac_compute_g_v1{};
        double t_g_v1[9]{};
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


    struct shared_global_ThreadVariables  {
        double * thread_data;

        double * g_arr_ptr(size_t id) {
            return thread_data + 0 + (id % 1);
        }
        double & g_arr(size_t id) {
            return thread_data[0 + (id % 1)];
        }
        double * g_w_ptr(size_t id) {
            return thread_data + 3 + (id % 1);
        }
        double & g_w(size_t id) {
            return thread_data[3 + (id % 1)];
        }
        double * g_v1_ptr(size_t id) {
            return thread_data + 4 + (id % 1);
        }
        double & g_v1(size_t id) {
            return thread_data[4 + (id % 1)];
        }

        shared_global_ThreadVariables(double * const thread_data) {
            this->thread_data = thread_data;
        }
    };


    static shared_global_Instance make_instance_shared_global(_nrn_mechanism_cache_range& _lmc) {
        return shared_global_Instance {
            _lmc.template fpfield_ptr<0>(),
            _lmc.template fpfield_ptr<1>(),
            _lmc.template fpfield_ptr<2>(),
            _lmc.template fpfield_ptr<3>(),
            _lmc.template fpfield_ptr<4>()
        };
    }


    static shared_global_NodeData make_node_data_shared_global(NrnThread& nt, Memb_list& _ml_arg) {
        return shared_global_NodeData {
            _ml_arg.nodeindices,
            nt.node_voltage_storage(),
            nt.node_d_storage(),
            nt.node_rhs_storage(),
            _ml_arg.nodecount
        };
    }


    static void nrn_alloc_shared_global(Prop* _prop) {
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
    inline double sum_arr_shared_global(_nrn_mechanism_cache_range& _lmc, shared_global_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, shared_global_ThreadVariables& _thread_vars, NrnThread* nt);
    inline int set_g_w_shared_global(_nrn_mechanism_cache_range& _lmc, shared_global_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, shared_global_ThreadVariables& _thread_vars, NrnThread* nt, double zz);
    inline int compute_g_v1_shared_global(_nrn_mechanism_cache_range& _lmc, shared_global_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, shared_global_ThreadVariables& _thread_vars, NrnThread* nt, double zz);
    void update_table_compute_g_v1_shared_global(_nrn_mechanism_cache_range& _lmc, shared_global_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, shared_global_ThreadVariables& _thread_vars, NrnThread* nt);
    static void _check_table_thread(Memb_list* _ml, size_t id, Datum* _ppvar, Datum* _thread, double* _globals, NrnThread* nt, int _type, const _nrn_model_sorted_token& _sorted_token)
{
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml, _type};
        auto inst = make_instance_shared_global(_lmc);
        auto _thread_vars = shared_global_ThreadVariables(_thread[0].get<double*>());
        update_table_compute_g_v1_shared_global(_lmc, inst, id, _ppvar, _thread, _thread_vars, nt);
    }


    /** connect global (scalar) variables to hoc -- */
    static DoubScal hoc_scalar_double[] = {
        {"usetable_shared_global", &shared_global_global.usetable},
        {"g_w_shared_global", &shared_global_global.thread_data[3]},
        {"g_v1_shared_global", &shared_global_global.thread_data[4]},
        {nullptr, nullptr}
    };


    /** connect global (array) variables to hoc -- */
    static DoubVec hoc_vector_double[] = {
        {"g_arr_shared_global", (shared_global_global.thread_data + 0), 3},
        {nullptr, nullptr, 0}
    };


    /* declaration of user functions */
    static void _hoc_set_g_w(void);
    static void _hoc_compute_g_v1(void);
    static void _hoc_sum_arr(void);
    static double _npy_set_g_w(Prop*);
    static double _npy_compute_g_v1(Prop*);
    static double _npy_sum_arr(Prop*);


    /* connect user functions to hoc names */
    static VoidFunc hoc_intfunc[] = {
        {"setdata_shared_global", _hoc_setdata},
        {"set_g_w_shared_global", _hoc_set_g_w},
        {"compute_g_v1_shared_global", _hoc_compute_g_v1},
        {"sum_arr_shared_global", _hoc_sum_arr},
        {nullptr, nullptr}
    };
    static NPyDirectMechFunc npy_direct_func_proc[] = {
        {"set_g_w", _npy_set_g_w},
        {"compute_g_v1", _npy_compute_g_v1},
        {"sum_arr", _npy_sum_arr},
        {nullptr, nullptr}
    };
    static void thread_mem_init(Datum* _thread)  {
        if(shared_global_global.thread_data_in_use) {
            _thread[0] = {neuron::container::do_not_search, new double[5]{}};
        }
        else {
            _thread[0] = {neuron::container::do_not_search, shared_global_global.thread_data};
            shared_global_global.thread_data_in_use = 1;
        }
    }
    static void thread_mem_cleanup(Datum* _thread)  {
        double * _thread_data_ptr = _thread[0].get<double*>();
        if(_thread_data_ptr == shared_global_global.thread_data) {
            shared_global_global.thread_data_in_use = 0;
        }
        else {
            delete[] _thread_data_ptr;
        }
    }
    static void _hoc_set_g_w(void) {
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
        auto inst = make_instance_shared_global(_lmc);
        auto _thread_vars = shared_global_ThreadVariables(_thread[0].get<double*>());
        _r = 1.;
        set_g_w_shared_global(_lmc, inst, id, _ppvar, _thread, _thread_vars, nt, *getarg(1));
        hoc_retpushx(_r);
    }
    static double _npy_set_g_w(Prop* _prop) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id{};
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_shared_global(_lmc);
        auto _thread_vars = shared_global_ThreadVariables(_thread[0].get<double*>());
        _r = 1.;
        set_g_w_shared_global(_lmc, inst, id, _ppvar, _thread, _thread_vars, nt, *getarg(1));
        return(_r);
    }
    static void _hoc_compute_g_v1(void) {
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
        auto inst = make_instance_shared_global(_lmc);
        auto _thread_vars = shared_global_ThreadVariables(_thread[0].get<double*>());
        update_table_compute_g_v1_shared_global(_lmc, inst, id, _ppvar, _thread, _thread_vars, nt);
        _r = 1.;
        compute_g_v1_shared_global(_lmc, inst, id, _ppvar, _thread, _thread_vars, nt, *getarg(1));
        hoc_retpushx(_r);
    }
    static double _npy_compute_g_v1(Prop* _prop) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id{};
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_shared_global(_lmc);
        auto _thread_vars = shared_global_ThreadVariables(_thread[0].get<double*>());
        update_table_compute_g_v1_shared_global(_lmc, inst, id, _ppvar, _thread, _thread_vars, nt);
        _r = 1.;
        compute_g_v1_shared_global(_lmc, inst, id, _ppvar, _thread, _thread_vars, nt, *getarg(1));
        return(_r);
    }
    static void _hoc_sum_arr(void) {
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
        auto inst = make_instance_shared_global(_lmc);
        auto _thread_vars = shared_global_ThreadVariables(_thread[0].get<double*>());
        _r = sum_arr_shared_global(_lmc, inst, id, _ppvar, _thread, _thread_vars, nt);
        hoc_retpushx(_r);
    }
    static double _npy_sum_arr(Prop* _prop) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id{};
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_shared_global(_lmc);
        auto _thread_vars = shared_global_ThreadVariables(_thread[0].get<double*>());
        _r = sum_arr_shared_global(_lmc, inst, id, _ppvar, _thread, _thread_vars, nt);
        return(_r);
    }


    inline int set_g_w_shared_global(_nrn_mechanism_cache_range& _lmc, shared_global_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, shared_global_ThreadVariables& _thread_vars, NrnThread* nt, double zz) {
        int ret_set_g_w = 0;
        auto v = inst.v_unused[id];
        _thread_vars.g_w(id) = zz;
        return ret_set_g_w;
    }


    inline static int f_compute_g_v1_shared_global(_nrn_mechanism_cache_range& _lmc, shared_global_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, shared_global_ThreadVariables& _thread_vars, NrnThread* nt, double zz) {
        int ret_f_compute_g_v1 = 0;
        auto v = inst.v_unused[id];
        _thread_vars.g_v1(id) = zz * zz;
        return ret_f_compute_g_v1;
    }


    void update_table_compute_g_v1_shared_global(_nrn_mechanism_cache_range& _lmc, shared_global_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, shared_global_ThreadVariables& _thread_vars, NrnThread* nt) {
        if (inst.global->usetable == 0) {
            return;
        }
        static bool make_table = true;
        if (make_table) {
            make_table = false;
            inst.global->tmin_compute_g_v1 = 3.0;
            double tmax = 4.0;
            double dx = (tmax-inst.global->tmin_compute_g_v1) / 8.;
            inst.global->mfac_compute_g_v1 = 1./dx;
            double x = inst.global->tmin_compute_g_v1;
            for (std::size_t i = 0; i < 9; x += dx, i++) {
                f_compute_g_v1_shared_global(_lmc, inst, id, _ppvar, _thread, _thread_vars, nt, x);
                inst.global->t_g_v1[i] = _thread_vars.g_v1(id);
            }
        }
    }


    inline int compute_g_v1_shared_global(_nrn_mechanism_cache_range& _lmc, shared_global_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, shared_global_ThreadVariables& _thread_vars, NrnThread* nt, double zz){
        if (inst.global->usetable == 0) {
            f_compute_g_v1_shared_global(_lmc, inst, id, _ppvar, _thread, _thread_vars, nt, zz);
            return 0;
        }
        double xi = inst.global->mfac_compute_g_v1 * (zz - inst.global->tmin_compute_g_v1);
        if (isnan(xi)) {
            _thread_vars.g_v1(id) = xi;
            return 0;
        }
        if (xi <= 0. || xi >= 8.) {
            int index = (xi <= 0.) ? 0 : 8;
            _thread_vars.g_v1(id) = inst.global->t_g_v1[index];
            return 0;
        }
        int i = int(xi);
        double theta = xi - double(i);
        _thread_vars.g_v1(id) = inst.global->t_g_v1[i] + theta*(inst.global->t_g_v1[i+1]-inst.global->t_g_v1[i]);
        return 0;
    }


    inline double sum_arr_shared_global(_nrn_mechanism_cache_range& _lmc, shared_global_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, shared_global_ThreadVariables& _thread_vars, NrnThread* nt) {
        double ret_sum_arr = 0.0;
        auto v = inst.v_unused[id];
        ret_sum_arr = (_thread_vars.g_arr_ptr(id))[static_cast<int>(0)] + (_thread_vars.g_arr_ptr(id))[static_cast<int>(1)] + (_thread_vars.g_arr_ptr(id))[static_cast<int>(2)];
        return ret_sum_arr;
    }


    void nrn_init_shared_global(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _type};
        auto inst = make_instance_shared_global(_lmc);
        auto node_data = make_node_data_shared_global(*nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* _thread = _ml_arg->_thread;
        auto _thread_vars = shared_global_ThreadVariables(_thread[0].get<double*>());
        for (int id = 0; id < nodecount; id++) {
            auto* _ppvar = _ml_arg->pdata[id];
            int node_id = node_data.nodeindices[id];
            auto v = node_data.node_voltages[node_id];
            inst.v_unused[id] = v;
            _thread_vars.g_w(id) = 48.0;
            _thread_vars.g_v1(id) = 0.0;
            (_thread_vars.g_arr_ptr(id))[static_cast<int>(0)] = 10.0 + inst.z[id];
            (_thread_vars.g_arr_ptr(id))[static_cast<int>(1)] = 10.1;
            (_thread_vars.g_arr_ptr(id))[static_cast<int>(2)] = 10.2;
            inst.y[id] = 10.0;
        }
    }


    inline double nrn_current_shared_global(_nrn_mechanism_cache_range& _lmc, NrnThread* nt, Datum* _ppvar, Datum* _thread, shared_global_ThreadVariables& _thread_vars, size_t id, shared_global_Instance& inst, shared_global_NodeData& node_data, double v) {
        double current = 0.0;
        if (nt->_t > 0.33) {
            _thread_vars.g_w(id) = sum_arr_shared_global(_lmc, inst, id, _ppvar, _thread, _thread_vars, nt);
        }
        if (nt->_t > 0.66) {
            set_g_w_shared_global(_lmc, inst, id, _ppvar, _thread, _thread_vars, nt, inst.z[id]);
            compute_g_v1_shared_global(_lmc, inst, id, _ppvar, _thread, _thread_vars, nt, inst.z[id]);
        }
        inst.y[id] = _thread_vars.g_w(id) + _thread_vars.g_v1(id);
        inst.il[id] = 0.0000001 * (v - 10.0);
        current += inst.il[id];
        return current;
    }


    /** update current */
    void nrn_cur_shared_global(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _type};
        auto inst = make_instance_shared_global(_lmc);
        auto node_data = make_node_data_shared_global(*nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* _thread = _ml_arg->_thread;
        auto _thread_vars = shared_global_ThreadVariables(_thread[0].get<double*>());
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_data.nodeindices[id];
            double v = node_data.node_voltages[node_id];
            auto* _ppvar = _ml_arg->pdata[id];
            double I1 = nrn_current_shared_global(_lmc, nt, _ppvar, _thread, _thread_vars, id, inst, node_data, v+0.001);
            double I0 = nrn_current_shared_global(_lmc, nt, _ppvar, _thread, _thread_vars, id, inst, node_data, v);
            double rhs = I0;
            double g = (I1-I0)/0.001;
            node_data.node_rhs[node_id] -= rhs;
            inst.g_unused[id] = g;
        }
    }


    void nrn_state_shared_global(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _type};
        auto inst = make_instance_shared_global(_lmc);
        auto node_data = make_node_data_shared_global(*nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* _thread = _ml_arg->_thread;
        auto _thread_vars = shared_global_ThreadVariables(_thread[0].get<double*>());
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_data.nodeindices[id];
            auto* _ppvar = _ml_arg->pdata[id];
            auto v = node_data.node_voltages[node_id];
        }
    }


    static void nrn_jacob_shared_global(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _type};
        auto inst = make_instance_shared_global(_lmc);
        auto node_data = make_node_data_shared_global(*nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
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
        _nrn_thread_table_reg(mech_type, _check_table_thread);
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
