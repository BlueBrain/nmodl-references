/*********************************************************
Model Name      : tbl
Filename        : table.mod
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

/* NEURON global macro definitions */
/* VECTORIZED */
#define NRN_VECTORIZED 1

static constexpr auto number_of_datum_variables = 0;
static constexpr auto number_of_floating_point_variables = 7;

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
        "tbl",
        0,
        "g_tbl",
        "i_tbl",
        "v1_tbl",
        "v2_tbl",
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
    struct tbl_Store {
        double usetable{1};
        double tmin_sigmoidal{};
        double mfac_sigmoidal{};
        double tmin_quadratic{};
        double mfac_quadratic{};
        double tmin_sinusoidal{};
        double mfac_sinusoidal{};
        double t_v1[801]{};
        double t_v2[801]{};
        double t_sig[156]{};
        double t_quadratic[501]{};
        double k{0.1};
        double d{-50};
        double c1{1};
        double c2{2};
    };
    static_assert(std::is_trivially_copy_constructible_v<tbl_Store>);
    static_assert(std::is_trivially_move_constructible_v<tbl_Store>);
    static_assert(std::is_trivially_copy_assignable_v<tbl_Store>);
    static_assert(std::is_trivially_move_assignable_v<tbl_Store>);
    static_assert(std::is_trivially_destructible_v<tbl_Store>);
    tbl_Store tbl_global;


    /** all mechanism instance variables and global variables */
    struct tbl_Instance  {
        double* g{};
        double* i{};
        double* v1{};
        double* v2{};
        double* sig{};
        double* v_unused{};
        double* g_unused{};
        tbl_Store* global{&tbl_global};
    };


    struct tbl_NodeData  {
        int const * nodeindices;
        double const * node_voltages;
        double * node_diagonal;
        double * node_rhs;
        int nodecount;
    };


    static tbl_Instance make_instance_tbl(_nrn_mechanism_cache_range& _lmc) {
        return tbl_Instance {
            _lmc.template fpfield_ptr<0>(),
            _lmc.template fpfield_ptr<1>(),
            _lmc.template fpfield_ptr<2>(),
            _lmc.template fpfield_ptr<3>(),
            _lmc.template fpfield_ptr<4>(),
            _lmc.template fpfield_ptr<5>(),
            _lmc.template fpfield_ptr<6>()
        };
    }


    static tbl_NodeData make_node_data_tbl(NrnThread& nt, Memb_list& _ml_arg) {
        return tbl_NodeData {
            _ml_arg.nodeindices,
            nt.node_voltage_storage(),
            nt.node_d_storage(),
            nt.node_rhs_storage(),
            _ml_arg.nodecount
        };
    }


    static void nrn_alloc_tbl(Prop* _prop) {
        Datum *_ppvar = nullptr;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const _iml = 0;
        assert(_nrn_mechanism_get_num_vars(_prop) == 7);
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
    inline double quadratic_tbl(_nrn_mechanism_cache_range& _lmc, tbl_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double x);
    inline int sigmoidal_tbl(_nrn_mechanism_cache_range& _lmc, tbl_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double v);
    inline int sinusoidal_tbl(_nrn_mechanism_cache_range& _lmc, tbl_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double x);
    void lazy_update_sigmoidal_tbl(_nrn_mechanism_cache_range& _lmc, tbl_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt);
    void lazy_update_quadratic_tbl(_nrn_mechanism_cache_range& _lmc, tbl_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt);
    void lazy_update_sinusoidal_tbl(_nrn_mechanism_cache_range& _lmc, tbl_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt);
    static void _check_table_thread(Memb_list* _ml, size_t id, Datum* _ppvar, Datum* _thread, double* _globals, NrnThread* nt, int _type, const _nrn_model_sorted_token& _sorted_token)
{
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml, _type};
        auto inst = make_instance_tbl(_lmc);
        lazy_update_sigmoidal_tbl(_lmc, inst, id, _ppvar, _thread, nt);
        lazy_update_quadratic_tbl(_lmc, inst, id, _ppvar, _thread, nt);
        lazy_update_sinusoidal_tbl(_lmc, inst, id, _ppvar, _thread, nt);
    }


    /** connect global (scalar) variables to hoc -- */
    static DoubScal hoc_scalar_double[] = {
        {"k_tbl", &tbl_global.k},
        {"d_tbl", &tbl_global.d},
        {"c1_tbl", &tbl_global.c1},
        {"c2_tbl", &tbl_global.c2},
        {"usetable_tbl", &tbl_global.usetable},
        {nullptr, nullptr}
    };


    /** connect global (array) variables to hoc -- */
    static DoubVec hoc_vector_double[] = {
        {nullptr, nullptr, 0}
    };


    /* declaration of user functions */
    static void _hoc_sigmoidal(void);
    static void _hoc_sinusoidal(void);
    static void _hoc_quadratic(void);
    static double _npy_sigmoidal(Prop*);
    static double _npy_sinusoidal(Prop*);
    static double _npy_quadratic(Prop*);


    /* connect user functions to hoc names */
    static VoidFunc hoc_intfunc[] = {
        {"setdata_tbl", _hoc_setdata},
        {"sigmoidal_tbl", _hoc_sigmoidal},
        {"sinusoidal_tbl", _hoc_sinusoidal},
        {"quadratic_tbl", _hoc_quadratic},
        {nullptr, nullptr}
    };
    static NPyDirectMechFunc npy_direct_func_proc[] = {
        {"sigmoidal", _npy_sigmoidal},
        {"sinusoidal", _npy_sinusoidal},
        {"quadratic", _npy_quadratic},
        {nullptr, nullptr}
    };
    static void _hoc_sigmoidal(void) {
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
        auto inst = make_instance_tbl(_lmc);
        lazy_update_sigmoidal_tbl(_lmc, inst, id, _ppvar, _thread, nt);
        _r = 1.;
        sigmoidal_tbl(_lmc, inst, id, _ppvar, _thread, nt, *getarg(1));
        hoc_retpushx(_r);
    }
    static double _npy_sigmoidal(Prop* _prop) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id{};
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_tbl(_lmc);
        lazy_update_sigmoidal_tbl(_lmc, inst, id, _ppvar, _thread, nt);
        _r = 1.;
        sigmoidal_tbl(_lmc, inst, id, _ppvar, _thread, nt, *getarg(1));
        return(_r);
    }
    static void _hoc_sinusoidal(void) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        if (!_prop_id) {
            hoc_execerror("No data for sinusoidal_tbl. Requires prior call to setdata_tbl and that the specified mechanism instance still be in existence.", NULL);
        }
        Prop* _local_prop = _extcall_prop;
        _nrn_mechanism_cache_instance _lmc{_local_prop};
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_tbl(_lmc);
        lazy_update_sinusoidal_tbl(_lmc, inst, id, _ppvar, _thread, nt);
        _r = 1.;
        sinusoidal_tbl(_lmc, inst, id, _ppvar, _thread, nt, *getarg(1));
        hoc_retpushx(_r);
    }
    static double _npy_sinusoidal(Prop* _prop) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id{};
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_tbl(_lmc);
        lazy_update_sinusoidal_tbl(_lmc, inst, id, _ppvar, _thread, nt);
        _r = 1.;
        sinusoidal_tbl(_lmc, inst, id, _ppvar, _thread, nt, *getarg(1));
        return(_r);
    }
    static void _hoc_quadratic(void) {
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
        auto inst = make_instance_tbl(_lmc);
        lazy_update_quadratic_tbl(_lmc, inst, id, _ppvar, _thread, nt);
        _r = quadratic_tbl(_lmc, inst, id, _ppvar, _thread, nt, *getarg(1));
        hoc_retpushx(_r);
    }
    static double _npy_quadratic(Prop* _prop) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id{};
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_tbl(_lmc);
        lazy_update_quadratic_tbl(_lmc, inst, id, _ppvar, _thread, nt);
        _r = quadratic_tbl(_lmc, inst, id, _ppvar, _thread, nt, *getarg(1));
        return(_r);
    }


    inline static int f_sigmoidal_tbl(_nrn_mechanism_cache_range& _lmc, tbl_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double v) {
        int ret_f_sigmoidal = 0;
        inst.sig[id] = 1.0 / (1.0 + exp(inst.global->k * (v - inst.global->d)));
        return ret_f_sigmoidal;
    }


    void lazy_update_sigmoidal_tbl(_nrn_mechanism_cache_range& _lmc, tbl_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt) {
        if (inst.global->usetable == 0) {
            return;
        }
        static bool make_table = true;
        static double save_k;
        static double save_d;
        if (save_k != inst.global->k) {
            make_table = true;
        }
        if (save_d != inst.global->d) {
            make_table = true;
        }
        if (make_table) {
            make_table = false;
            inst.global->tmin_sigmoidal =  -127.0;
            double tmax = 128.0;
            double dx = (tmax-inst.global->tmin_sigmoidal) / 155.;
            inst.global->mfac_sigmoidal = 1./dx;
            double x = inst.global->tmin_sigmoidal;
            for (std::size_t i = 0; i < 156; x += dx, i++) {
                f_sigmoidal_tbl(_lmc, inst, id, _ppvar, _thread, nt, x);
                inst.global->t_sig[i] = inst.sig[id];
            }
            save_k = inst.global->k;
            save_d = inst.global->d;
        }
    }


    inline int sigmoidal_tbl(_nrn_mechanism_cache_range& _lmc, tbl_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double v){
        if (inst.global->usetable == 0) {
            f_sigmoidal_tbl(_lmc, inst, id, _ppvar, _thread, nt, v);
            return 0;
        }
        double xi = inst.global->mfac_sigmoidal * (v - inst.global->tmin_sigmoidal);
        if (isnan(xi)) {
            inst.sig[id] = xi;
            return 0;
        }
        if (xi <= 0. || xi >= 155.) {
            int index = (xi <= 0.) ? 0 : 155;
            inst.sig[id] = inst.global->t_sig[index];
            return 0;
        }
        int i = int(xi);
        double theta = xi - double(i);
        inst.sig[id] = inst.global->t_sig[i] + theta*(inst.global->t_sig[i+1]-inst.global->t_sig[i]);
        return 0;
    }


    inline static int f_sinusoidal_tbl(_nrn_mechanism_cache_range& _lmc, tbl_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double x) {
        int ret_f_sinusoidal = 0;
        auto v = inst.v_unused[id];
        inst.v1[id] = sin(inst.global->c1 * x) + 2.0;
        inst.v2[id] = cos(inst.global->c2 * x) + 2.0;
        return ret_f_sinusoidal;
    }


    void lazy_update_sinusoidal_tbl(_nrn_mechanism_cache_range& _lmc, tbl_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt) {
        if (inst.global->usetable == 0) {
            return;
        }
        static bool make_table = true;
        static double save_c1;
        static double save_c2;
        if (save_c1 != inst.global->c1) {
            make_table = true;
        }
        if (save_c2 != inst.global->c2) {
            make_table = true;
        }
        if (make_table) {
            make_table = false;
            inst.global->tmin_sinusoidal =  -4.0;
            double tmax = 6.0;
            double dx = (tmax-inst.global->tmin_sinusoidal) / 800.;
            inst.global->mfac_sinusoidal = 1./dx;
            double x = inst.global->tmin_sinusoidal;
            for (std::size_t i = 0; i < 801; x += dx, i++) {
                f_sinusoidal_tbl(_lmc, inst, id, _ppvar, _thread, nt, x);
                inst.global->t_v1[i] = inst.v1[id];
                inst.global->t_v2[i] = inst.v2[id];
            }
            save_c1 = inst.global->c1;
            save_c2 = inst.global->c2;
        }
    }


    inline int sinusoidal_tbl(_nrn_mechanism_cache_range& _lmc, tbl_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double x){
        if (inst.global->usetable == 0) {
            f_sinusoidal_tbl(_lmc, inst, id, _ppvar, _thread, nt, x);
            return 0;
        }
        double xi = inst.global->mfac_sinusoidal * (x - inst.global->tmin_sinusoidal);
        if (isnan(xi)) {
            inst.v1[id] = xi;
            inst.v2[id] = xi;
            return 0;
        }
        if (xi <= 0. || xi >= 800.) {
            int index = (xi <= 0.) ? 0 : 800;
            inst.v1[id] = inst.global->t_v1[index];
            inst.v2[id] = inst.global->t_v2[index];
            return 0;
        }
        int i = int(xi);
        double theta = xi - double(i);
        inst.v1[id] = inst.global->t_v1[i] + theta*(inst.global->t_v1[i+1]-inst.global->t_v1[i]);
        inst.v2[id] = inst.global->t_v2[i] + theta*(inst.global->t_v2[i+1]-inst.global->t_v2[i]);
        return 0;
    }


    inline static double f_quadratic_tbl(_nrn_mechanism_cache_range& _lmc, tbl_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double x) {
        double ret_f_quadratic = 0.0;
        auto v = inst.v_unused[id];
        ret_f_quadratic = inst.global->c1 * x * x + inst.global->c2;
        return ret_f_quadratic;
    }


    void lazy_update_quadratic_tbl(_nrn_mechanism_cache_range& _lmc, tbl_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt) {
        if (inst.global->usetable == 0) {
            return;
        }
        static bool make_table = true;
        static double save_c1;
        static double save_c2;
        if (save_c1 != inst.global->c1) {
            make_table = true;
        }
        if (save_c2 != inst.global->c2) {
            make_table = true;
        }
        if (make_table) {
            make_table = false;
            inst.global->tmin_quadratic =  -3.0;
            double tmax = 5.0;
            double dx = (tmax-inst.global->tmin_quadratic) / 500.;
            inst.global->mfac_quadratic = 1./dx;
            double x = inst.global->tmin_quadratic;
            for (std::size_t i = 0; i < 501; x += dx, i++) {
                inst.global->t_quadratic[i] = f_quadratic_tbl(_lmc, inst, id, _ppvar, _thread, nt, x);
            }
            save_c1 = inst.global->c1;
            save_c2 = inst.global->c2;
        }
    }


    inline double quadratic_tbl(_nrn_mechanism_cache_range& _lmc, tbl_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double x){
        if (inst.global->usetable == 0) {
            return f_quadratic_tbl(_lmc, inst, id, _ppvar, _thread, nt, x);
        }
        double xi = inst.global->mfac_quadratic * (x - inst.global->tmin_quadratic);
        if (isnan(xi)) {
            return xi;
        }
        if (xi <= 0. || xi >= 500.) {
            int index = (xi <= 0.) ? 0 : 500;
            return inst.global->t_quadratic[index];
        }
        int i = int(xi);
        double theta = xi - double(i);
        return inst.global->t_quadratic[i] + theta * (inst.global->t_quadratic[i+1] - inst.global->t_quadratic[i]);
    }


    void nrn_init_tbl(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _type};
        auto inst = make_instance_tbl(_lmc);
        auto node_data = make_node_data_tbl(*nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* _thread = _ml_arg->_thread;
        for (int id = 0; id < nodecount; id++) {
            auto* _ppvar = _ml_arg->pdata[id];
            int node_id = node_data.nodeindices[id];
            auto v = node_data.node_voltages[node_id];
            inst.v_unused[id] = v;
        }
    }


    inline double nrn_current_tbl(_nrn_mechanism_cache_range& _lmc, NrnThread* nt, Datum* _ppvar, Datum* _thread, size_t id, tbl_Instance& inst, tbl_NodeData& node_data, double v) {
        double current = 0.0;
        sigmoidal_tbl(_lmc, inst, id, _ppvar, _thread, nt, v);
        inst.g[id] = 0.001 * inst.sig[id];
        inst.i[id] = inst.g[id] * (v - 30.0);
        current += inst.i[id];
        return current;
    }


    /** update current */
    void nrn_cur_tbl(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _type};
        auto inst = make_instance_tbl(_lmc);
        auto node_data = make_node_data_tbl(*nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* _thread = _ml_arg->_thread;
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_data.nodeindices[id];
            double v = node_data.node_voltages[node_id];
            auto* _ppvar = _ml_arg->pdata[id];
            double I1 = nrn_current_tbl(_lmc, nt, _ppvar, _thread, id, inst, node_data, v+0.001);
            double I0 = nrn_current_tbl(_lmc, nt, _ppvar, _thread, id, inst, node_data, v);
            double rhs = I0;
            double g = (I1-I0)/0.001;
            node_data.node_rhs[node_id] -= rhs;
            inst.g_unused[id] = g;
        }
    }


    void nrn_state_tbl(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _type};
        auto inst = make_instance_tbl(_lmc);
        auto node_data = make_node_data_tbl(*nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* _thread = _ml_arg->_thread;
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_data.nodeindices[id];
            auto* _ppvar = _ml_arg->pdata[id];
            auto v = node_data.node_voltages[node_id];
        }
    }


    static void nrn_jacob_tbl(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _type};
        auto inst = make_instance_tbl(_lmc);
        auto node_data = make_node_data_tbl(*nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_data.nodeindices[id];
            node_data.node_diagonal[node_id] += inst.g_unused[id];
        }
    }


    static void _initlists() {
    }


    /** register channel with the simulator */
    extern "C" void _table_reg() {
        _initlists();

        register_mech(mechanism_info, nrn_alloc_tbl, nrn_cur_tbl, nrn_jacob_tbl, nrn_state_tbl, nrn_init_tbl, hoc_nrnpointerindex, 1);

        mech_type = nrn_get_mechtype(mechanism_info[1]);
        _nrn_thread_table_reg(mech_type, _check_table_thread);
        _nrn_mechanism_register_data_fields(mech_type,
            _nrn_mechanism_field<double>{"g"} /* 0 */,
            _nrn_mechanism_field<double>{"i"} /* 1 */,
            _nrn_mechanism_field<double>{"v1"} /* 2 */,
            _nrn_mechanism_field<double>{"v2"} /* 3 */,
            _nrn_mechanism_field<double>{"sig"} /* 4 */,
            _nrn_mechanism_field<double>{"v_unused"} /* 5 */,
            _nrn_mechanism_field<double>{"g_unused"} /* 6 */
        );

        hoc_register_prop_size(mech_type, 7, 0);
        hoc_register_var(hoc_scalar_double, hoc_vector_double, hoc_intfunc);
        hoc_register_npy_direct(mech_type, npy_direct_func_proc);
    }
}
