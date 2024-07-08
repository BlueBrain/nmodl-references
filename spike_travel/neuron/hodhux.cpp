/*********************************************************
Model Name      : hodhux
Filename        : hodhux.mod
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

static constexpr auto number_of_datum_variables = 6;
static constexpr auto number_of_floating_point_variables = 23;

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
extern double celsius;


namespace neuron {
    #ifndef NRN_PRCELLSTATE
    #define NRN_PRCELLSTATE 0
    #endif


    /** channel information */
    static const char *mechanism_info[] = {
        "7.7.0",
        "hodhux",
        "gnabar_hodhux",
        "gkbar_hodhux",
        "gl_hodhux",
        "el_hodhux",
        0,
        "il_hodhux",
        "minf_hodhux",
        "hinf_hodhux",
        "ninf_hodhux",
        "mexp_hodhux",
        "hexp_hodhux",
        "nexp_hodhux",
        0,
        "m_hodhux",
        "h_hodhux",
        "n_hodhux",
        0,
        0
    };


    /* NEURON global variables */
    static Symbol* _na_sym;
    static Symbol* _k_sym;
    static int mech_type;
    static Prop* _extcall_prop;
    /* _prop_id kind of shadows _extcall_prop to allow validity checking. */
    static _nrn_non_owning_id_without_container _prop_id{};
    static int hoc_nrnpointerindex = -1;
    static _nrn_mechanism_std_vector<Datum> _extcall_thread;


    /** all global variables */
    struct hodhux_Store {
        double m0{};
        double h0{};
        double n0{};
    };
    static_assert(std::is_trivially_copy_constructible_v<hodhux_Store>);
    static_assert(std::is_trivially_move_constructible_v<hodhux_Store>);
    static_assert(std::is_trivially_copy_assignable_v<hodhux_Store>);
    static_assert(std::is_trivially_move_assignable_v<hodhux_Store>);
    static_assert(std::is_trivially_destructible_v<hodhux_Store>);
    hodhux_Store hodhux_global;


    /** all mechanism instance variables and global variables */
    struct hodhux_Instance  {
        double* celsius{&::celsius};
        double* gnabar{};
        double* gkbar{};
        double* gl{};
        double* el{};
        double* il{};
        double* minf{};
        double* hinf{};
        double* ninf{};
        double* mexp{};
        double* hexp{};
        double* nexp{};
        double* m{};
        double* h{};
        double* n{};
        double* ena{};
        double* ek{};
        double* Dm{};
        double* Dh{};
        double* Dn{};
        double* ina{};
        double* ik{};
        double* v_unused{};
        double* g_unused{};
        const double* const* ion_ena{};
        double* const* ion_ina{};
        double* const* ion_dinadv{};
        const double* const* ion_ek{};
        double* const* ion_ik{};
        double* const* ion_dikdv{};
        hodhux_Store* global{&hodhux_global};
    };


    struct hodhux_NodeData  {
        int const * nodeindices;
        double const * node_voltages;
        double * node_diagonal;
        double * node_rhs;
        int nodecount;
    };


    static hodhux_Instance make_instance_hodhux(_nrn_mechanism_cache_range& _lmc) {
        return hodhux_Instance {
            &::celsius,
            _lmc.template fpfield_ptr<0>(),
            _lmc.template fpfield_ptr<1>(),
            _lmc.template fpfield_ptr<2>(),
            _lmc.template fpfield_ptr<3>(),
            _lmc.template fpfield_ptr<4>(),
            _lmc.template fpfield_ptr<5>(),
            _lmc.template fpfield_ptr<6>(),
            _lmc.template fpfield_ptr<7>(),
            _lmc.template fpfield_ptr<8>(),
            _lmc.template fpfield_ptr<9>(),
            _lmc.template fpfield_ptr<10>(),
            _lmc.template fpfield_ptr<11>(),
            _lmc.template fpfield_ptr<12>(),
            _lmc.template fpfield_ptr<13>(),
            _lmc.template fpfield_ptr<14>(),
            _lmc.template fpfield_ptr<15>(),
            _lmc.template fpfield_ptr<16>(),
            _lmc.template fpfield_ptr<17>(),
            _lmc.template fpfield_ptr<18>(),
            _lmc.template fpfield_ptr<19>(),
            _lmc.template fpfield_ptr<20>(),
            _lmc.template fpfield_ptr<21>(),
            _lmc.template fpfield_ptr<22>(),
            _lmc.template dptr_field_ptr<0>(),
            _lmc.template dptr_field_ptr<1>(),
            _lmc.template dptr_field_ptr<2>(),
            _lmc.template dptr_field_ptr<3>(),
            _lmc.template dptr_field_ptr<4>(),
            _lmc.template dptr_field_ptr<5>()
        };
    }


    static hodhux_NodeData make_node_data_hodhux(NrnThread& nt, Memb_list& _ml_arg) {
        return hodhux_NodeData {
            _ml_arg.nodeindices,
            nt.node_voltage_storage(),
            nt.node_d_storage(),
            nt.node_rhs_storage(),
            _ml_arg.nodecount
        };
    }


    static void nrn_alloc_hodhux(Prop* _prop) {
        Prop *prop_ion{};
        Datum *_ppvar{};
        _ppvar = nrn_prop_datum_alloc(mech_type, 6, _prop);
        _nrn_mechanism_access_dparam(_prop) = _ppvar;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const _iml{};
        assert(_nrn_mechanism_get_num_vars(_prop) == 23);
        /*initialize range parameters*/
        _lmc.template fpfield<0>(_iml) = 0.12; /* gnabar */
        _lmc.template fpfield<1>(_iml) = 0.036; /* gkbar */
        _lmc.template fpfield<2>(_iml) = 0.0003; /* gl */
        _lmc.template fpfield<3>(_iml) = -54.3; /* el */
        _nrn_mechanism_access_dparam(_prop) = _ppvar;
        Symbol * na_sym = hoc_lookup("na_ion");
        Prop * na_prop = need_memb(na_sym);
        _ppvar[0] = _nrn_mechanism_get_param_handle(na_prop, 0);
        _ppvar[1] = _nrn_mechanism_get_param_handle(na_prop, 3);
        _ppvar[2] = _nrn_mechanism_get_param_handle(na_prop, 4);
        Symbol * k_sym = hoc_lookup("k_ion");
        Prop * k_prop = need_memb(k_sym);
        _ppvar[3] = _nrn_mechanism_get_param_handle(k_prop, 0);
        _ppvar[4] = _nrn_mechanism_get_param_handle(k_prop, 3);
        _ppvar[5] = _nrn_mechanism_get_param_handle(k_prop, 4);
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
    inline double vtrap_hodhux(_nrn_mechanism_cache_range& _lmc, hodhux_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double x, double y);
    inline int states_hodhux(_nrn_mechanism_cache_range& _lmc, hodhux_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt);
    inline int rates_hodhux(_nrn_mechanism_cache_range& _lmc, hodhux_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double v);


    /** connect global (scalar) variables to hoc -- */
    static DoubScal hoc_scalar_double[] = {
        {nullptr, nullptr}
    };


    /** connect global (array) variables to hoc -- */
    static DoubVec hoc_vector_double[] = {
        {nullptr, nullptr, 0}
    };


    /* declaration of user functions */
    static void _hoc_states(void);
    static void _hoc_rates(void);
    static void _hoc_vtrap(void);
    static double _npy_states(Prop*);
    static double _npy_rates(Prop*);
    static double _npy_vtrap(Prop*);


    /* connect user functions to hoc names */
    static VoidFunc hoc_intfunc[] = {
        {"setdata_hodhux", _hoc_setdata},
        {"states_hodhux", _hoc_states},
        {"rates_hodhux", _hoc_rates},
        {"vtrap_hodhux", _hoc_vtrap},
        {nullptr, nullptr}
    };
    static NPyDirectMechFunc npy_direct_func_proc[] = {
        {"states", _npy_states},
        {"rates", _npy_rates},
        {"vtrap", _npy_vtrap},
        {nullptr, nullptr}
    };
    static void _hoc_states(void) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        if (!_prop_id) {
            hoc_execerror("No data for states_hodhux. Requires prior call to setdata_hodhux and that the specified mechanism instance still be in existence.", NULL);
        }
        Prop* _local_prop = _extcall_prop;
        _nrn_mechanism_cache_instance _lmc{_local_prop};
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_hodhux(_lmc);
        _r = 1.;
        states_hodhux(_lmc, inst, id, _ppvar, _thread, nt);
        hoc_retpushx(_r);
    }
    static double _npy_states(Prop* _prop) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id{};
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_hodhux(_lmc);
        _r = 1.;
        states_hodhux(_lmc, inst, id, _ppvar, _thread, nt);
        return(_r);
    }
    static void _hoc_rates(void) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        if (!_prop_id) {
            hoc_execerror("No data for rates_hodhux. Requires prior call to setdata_hodhux and that the specified mechanism instance still be in existence.", NULL);
        }
        Prop* _local_prop = _extcall_prop;
        _nrn_mechanism_cache_instance _lmc{_local_prop};
        size_t const id{};
        _ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_hodhux(_lmc);
        _r = 1.;
        rates_hodhux(_lmc, inst, id, _ppvar, _thread, nt, *getarg(1));
        hoc_retpushx(_r);
    }
    static double _npy_rates(Prop* _prop) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id{};
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_hodhux(_lmc);
        _r = 1.;
        rates_hodhux(_lmc, inst, id, _ppvar, _thread, nt, *getarg(1));
        return(_r);
    }
    static void _hoc_vtrap(void) {
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
        auto inst = make_instance_hodhux(_lmc);
        _r = vtrap_hodhux(_lmc, inst, id, _ppvar, _thread, nt, *getarg(1), *getarg(2));
        hoc_retpushx(_r);
    }
    static double _npy_vtrap(Prop* _prop) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const id{};
        _ppvar = _nrn_mechanism_access_dparam(_prop);
        _thread = _extcall_thread.data();
        nt = nrn_threads;
        auto inst = make_instance_hodhux(_lmc);
        _r = vtrap_hodhux(_lmc, inst, id, _ppvar, _thread, nt, *getarg(1), *getarg(2));
        return(_r);
    }


    inline int states_hodhux(_nrn_mechanism_cache_range& _lmc, hodhux_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt) {
        int ret_states = 0;
        auto v = inst.v_unused[id];
        rates_hodhux(_lmc, inst, id, _ppvar, _thread, nt, v);
        inst.m[id] = inst.m[id] + inst.mexp[id] * (inst.minf[id] - inst.m[id]);
        inst.h[id] = inst.h[id] + inst.hexp[id] * (inst.hinf[id] - inst.h[id]);
        inst.n[id] = inst.n[id] + inst.nexp[id] * (inst.ninf[id] - inst.n[id]);
        return ret_states;
    }


    inline int rates_hodhux(_nrn_mechanism_cache_range& _lmc, hodhux_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double v) {
        int ret_rates = 0;
        double q10, tinc, alpha, beta, sum;
        q10 = pow(3.0, ((*(inst.celsius) - 6.3) / 10.0));
        tinc =  -nt->_dt * q10;
        alpha = .1 * vtrap_hodhux(_lmc, inst, id, _ppvar, _thread, nt,  -(v + 40.0), 10.0);
        beta = 4.0 * exp( -(v + 65.0) / 18.0);
        sum = alpha + beta;
        inst.minf[id] = alpha / sum;
        inst.mexp[id] = 1.0 - exp(tinc * sum);
        alpha = .07 * exp( -(v + 65.0) / 20.0);
        beta = 1.0 / (exp( -(v + 35.0) / 10.0) + 1.0);
        sum = alpha + beta;
        inst.hinf[id] = alpha / sum;
        inst.hexp[id] = 1.0 - exp(tinc * sum);
        alpha = .01 * vtrap_hodhux(_lmc, inst, id, _ppvar, _thread, nt,  -(v + 55.0), 10.0);
        beta = .125 * exp( -(v + 65.0) / 80.0);
        sum = alpha + beta;
        inst.ninf[id] = alpha / sum;
        inst.nexp[id] = 1.0 - exp(tinc * sum);
        return ret_rates;
    }


    inline double vtrap_hodhux(_nrn_mechanism_cache_range& _lmc, hodhux_Instance& inst, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double x, double y) {
        double ret_vtrap = 0.0;
        auto v = inst.v_unused[id];
        if (fabs(x / y) < 1e-6) {
            ret_vtrap = y * (1.0 - x / y / 2.0);
        } else {
            ret_vtrap = x / (exp(x / y) - 1.0);
        }
        return ret_vtrap;
    }


    void nrn_init_hodhux(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _type};
        auto inst = make_instance_hodhux(_lmc);
        auto node_data = make_node_data_hodhux(*nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* _thread = _ml_arg->_thread;
        for (int id = 0; id < nodecount; id++) {
            auto* _ppvar = _ml_arg->pdata[id];
            int node_id = node_data.nodeindices[id];
            auto v = node_data.node_voltages[node_id];
            inst.v_unused[id] = v;
            inst.ena[id] = (*inst.ion_ena[id]);
            inst.ek[id] = (*inst.ion_ek[id]);
            rates_hodhux(_lmc, inst, id, _ppvar, _thread, nt, v);
            inst.m[id] = inst.minf[id];
            inst.h[id] = inst.hinf[id];
            inst.n[id] = inst.ninf[id];
        }
    }


    inline double nrn_current_hodhux(_nrn_mechanism_cache_range& _lmc, NrnThread* nt, Datum* _ppvar, Datum* _thread, size_t id, hodhux_Instance& inst, hodhux_NodeData& node_data, double v) {
        double current = 0.0;
        inst.ina[id] = inst.gnabar[id] * inst.m[id] * inst.m[id] * inst.m[id] * inst.h[id] * (v - inst.ena[id]);
        inst.ik[id] = inst.gkbar[id] * inst.n[id] * inst.n[id] * inst.n[id] * inst.n[id] * (v - inst.ek[id]);
        inst.il[id] = inst.gl[id] * (v - inst.el[id]);
        current += inst.il[id];
        current += inst.ina[id];
        current += inst.ik[id];
        return current;
    }


    /** update current */
    void nrn_cur_hodhux(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _type};
        auto inst = make_instance_hodhux(_lmc);
        auto node_data = make_node_data_hodhux(*nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* _thread = _ml_arg->_thread;
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_data.nodeindices[id];
            double v = node_data.node_voltages[node_id];
            auto* _ppvar = _ml_arg->pdata[id];
            inst.ena[id] = (*inst.ion_ena[id]);
            inst.ek[id] = (*inst.ion_ek[id]);
            double I1 = nrn_current_hodhux(_lmc, nt, _ppvar, _thread, id, inst, node_data, v+0.001);
            double dina = inst.ina[id];
            double dik = inst.ik[id];
            double I0 = nrn_current_hodhux(_lmc, nt, _ppvar, _thread, id, inst, node_data, v);
            double rhs = I0;
            double g = (I1-I0)/0.001;
            (*inst.ion_dinadv[id]) += (dina-inst.ina[id])/0.001;
            (*inst.ion_dikdv[id]) += (dik-inst.ik[id])/0.001;
            (*inst.ion_ina[id]) += inst.ina[id];
            (*inst.ion_ik[id]) += inst.ik[id];
            node_data.node_rhs[node_id] -= rhs;
            inst.g_unused[id] = g;
        }
    }


    void nrn_state_hodhux(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _type};
        auto inst = make_instance_hodhux(_lmc);
        auto node_data = make_node_data_hodhux(*nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        auto* _thread = _ml_arg->_thread;
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_data.nodeindices[id];
            auto* _ppvar = _ml_arg->pdata[id];
            auto v = node_data.node_voltages[node_id];
            inst.ena[id] = (*inst.ion_ena[id]);
            inst.ek[id] = (*inst.ion_ek[id]);
            rates_hodhux(_lmc, inst, id, _ppvar, _thread, nt, v);
            inst.m[id] = inst.m[id] + inst.mexp[id] * (inst.minf[id] - inst.m[id]);
            inst.h[id] = inst.h[id] + inst.hexp[id] * (inst.hinf[id] - inst.h[id]);
            inst.n[id] = inst.n[id] + inst.nexp[id] * (inst.ninf[id] - inst.n[id]);
        }
    }


    static void nrn_jacob_hodhux(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _type};
        auto inst = make_instance_hodhux(_lmc);
        auto node_data = make_node_data_hodhux(*nt, *_ml_arg);
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_data.nodeindices[id];
            node_data.node_diagonal[node_id] += inst.g_unused[id];
        }
    }


    static void _initlists() {
    }


    /** register channel with the simulator */
    extern "C" void _hodhux_reg() {
        _initlists();

        ion_reg("na", -10000.);
        ion_reg("k", -10000.);

        _na_sym = hoc_lookup("na_ion");
        _k_sym = hoc_lookup("k_ion");

        register_mech(mechanism_info, nrn_alloc_hodhux, nrn_cur_hodhux, nrn_jacob_hodhux, nrn_state_hodhux, nrn_init_hodhux, hoc_nrnpointerindex, 1);

        mech_type = nrn_get_mechtype(mechanism_info[1]);
        _nrn_mechanism_register_data_fields(mech_type,
            _nrn_mechanism_field<double>{"gnabar"} /* 0 */,
            _nrn_mechanism_field<double>{"gkbar"} /* 1 */,
            _nrn_mechanism_field<double>{"gl"} /* 2 */,
            _nrn_mechanism_field<double>{"el"} /* 3 */,
            _nrn_mechanism_field<double>{"il"} /* 4 */,
            _nrn_mechanism_field<double>{"minf"} /* 5 */,
            _nrn_mechanism_field<double>{"hinf"} /* 6 */,
            _nrn_mechanism_field<double>{"ninf"} /* 7 */,
            _nrn_mechanism_field<double>{"mexp"} /* 8 */,
            _nrn_mechanism_field<double>{"hexp"} /* 9 */,
            _nrn_mechanism_field<double>{"nexp"} /* 10 */,
            _nrn_mechanism_field<double>{"m"} /* 11 */,
            _nrn_mechanism_field<double>{"h"} /* 12 */,
            _nrn_mechanism_field<double>{"n"} /* 13 */,
            _nrn_mechanism_field<double>{"ena"} /* 14 */,
            _nrn_mechanism_field<double>{"ek"} /* 15 */,
            _nrn_mechanism_field<double>{"Dm"} /* 16 */,
            _nrn_mechanism_field<double>{"Dh"} /* 17 */,
            _nrn_mechanism_field<double>{"Dn"} /* 18 */,
            _nrn_mechanism_field<double>{"ina"} /* 19 */,
            _nrn_mechanism_field<double>{"ik"} /* 20 */,
            _nrn_mechanism_field<double>{"v_unused"} /* 21 */,
            _nrn_mechanism_field<double>{"g_unused"} /* 22 */,
            _nrn_mechanism_field<double*>{"ion_ena", "na_ion"} /* 0 */,
            _nrn_mechanism_field<double*>{"ion_ina", "na_ion"} /* 1 */,
            _nrn_mechanism_field<double*>{"ion_dinadv", "na_ion"} /* 2 */,
            _nrn_mechanism_field<double*>{"ion_ek", "k_ion"} /* 3 */,
            _nrn_mechanism_field<double*>{"ion_ik", "k_ion"} /* 4 */,
            _nrn_mechanism_field<double*>{"ion_dikdv", "k_ion"} /* 5 */
        );

        hoc_register_prop_size(mech_type, 23, 6);
        hoc_register_dparam_semantics(mech_type, 0, "na_ion");
        hoc_register_dparam_semantics(mech_type, 1, "na_ion");
        hoc_register_dparam_semantics(mech_type, 2, "na_ion");
        hoc_register_dparam_semantics(mech_type, 3, "k_ion");
        hoc_register_dparam_semantics(mech_type, 4, "k_ion");
        hoc_register_dparam_semantics(mech_type, 5, "k_ion");
        hoc_register_var(hoc_scalar_double, hoc_vector_double, hoc_intfunc);
        hoc_register_npy_direct(mech_type, npy_direct_func_proc);
    }
}
