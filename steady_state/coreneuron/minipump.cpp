/*********************************************************
Model Name      : minipump
Filename        : minipump.mod
NMODL Version   : 7.7.0
Vectorized      : true
Threadsafe      : true
Created         : DATE
Simulator       : CoreNEURON
Backend         : C++ (api-compatibility)
NMODL Compiler  : VERSION
*********************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <coreneuron/gpu/nrn_acc_manager.hpp>
#include <coreneuron/mechanism/mech/mod2c_core_thread.hpp>
#include <coreneuron/mechanism/register_mech.hpp>
#include <coreneuron/nrnconf.h>
#include <coreneuron/nrniv/nrniv_decl.h>
#include <coreneuron/sim/multicore.hpp>
#include <coreneuron/sim/scopmath/newton_thread.hpp>
#include <coreneuron/utils/ivocvect.hpp>
#include <coreneuron/utils/nrnoc_aux.hpp>
#include <coreneuron/utils/randoms/nrnran123.h>

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
bool is_converged(const Eigen::Matrix<double, N, 1>& X,
                  const Eigen::Matrix<double, N, N>& J,
                  const Eigen::Matrix<double, N, 1>& F,
                  double eps) {
    for (Eigen::Index i = 0; i < N; ++i) {
        double square_error = J(i, Eigen::all).cwiseAbs2() * (eps * X).cwiseAbs2();
        if (F(i) * F(i) > square_error) {
            return false;
        }
    }
    return true;
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
    // Vector to store result of function F(X):
    Eigen::Matrix<double, N, 1> F;
    // Matrix to store Jacobian of F(X):
    Eigen::Matrix<double, N, N> J;
    // Solver iteration count:
    int iter = -1;
    while (++iter < max_iter) {
        // calculate F, J from X using user-supplied functor
        functor(X, F, J);
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
    Eigen::Matrix<double, N, N> J, J_inv;
    int iter = -1;
    while (++iter < max_iter) {
        functor(X, F, J);
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



namespace coreneuron {
    #ifndef NRN_PRCELLSTATE
    #define NRN_PRCELLSTATE 0
    #endif


    /** channel information */
    static const char *mechanism_info[] = {
        "7.7.0",
        "minipump",
        0,
        0,
        "X_minipump",
        "Y_minipump",
        "Z_minipump",
        0,
        0
    };


    /** all global variables */
    struct minipump_Store {
        double X0{};
        double Y0{};
        double Z0{};
        int reset{};
        int mech_type{};
        double volA{1e+09};
        double volB{1e+09};
        double volC{13};
        double kf{3};
        double kb{4};
        double run_steady_state{0};
        int slist1[3]{0, 1, 2};
        int dlist1[3]{3, 4, 5};
    };
    static_assert(std::is_trivially_copy_constructible_v<minipump_Store>);
    static_assert(std::is_trivially_move_constructible_v<minipump_Store>);
    static_assert(std::is_trivially_copy_assignable_v<minipump_Store>);
    static_assert(std::is_trivially_move_assignable_v<minipump_Store>);
    static_assert(std::is_trivially_destructible_v<minipump_Store>);
    minipump_Store minipump_global;


    /** all mechanism instance variables and global variables */
    struct minipump_Instance  {
        double* X{};
        double* Y{};
        double* Z{};
        double* DX{};
        double* DY{};
        double* DZ{};
        double* v_unused{};
        double* g_unused{};
        minipump_Store* global{&minipump_global};
    };


    /** connect global (scalar) variables to hoc -- */
    static DoubScal hoc_scalar_double[] = {
        {"volA_minipump", &minipump_global.volA},
        {"volB_minipump", &minipump_global.volB},
        {"volC_minipump", &minipump_global.volC},
        {"kf_minipump", &minipump_global.kf},
        {"kb_minipump", &minipump_global.kb},
        {"run_steady_state_minipump", &minipump_global.run_steady_state},
        {nullptr, nullptr}
    };


    /** connect global (array) variables to hoc -- */
    static DoubVec hoc_vector_double[] = {
        {nullptr, nullptr, 0}
    };


    static inline int first_pointer_var_index() {
        return -1;
    }


    static inline int first_random_var_index() {
        return -1;
    }


    static inline int float_variables_size() {
        return 8;
    }


    static inline int int_variables_size() {
        return 0;
    }


    static inline int get_mech_type() {
        return minipump_global.mech_type;
    }


    static inline Memb_list* get_memb_list(NrnThread* nt) {
        if (!nt->_ml_list) {
            return nullptr;
        }
        return nt->_ml_list[get_mech_type()];
    }


    static inline void* mem_alloc(size_t num, size_t size, size_t alignment = 64) {
        size_t aligned_size = ((num*size + alignment - 1) / alignment) * alignment;
        void* ptr = aligned_alloc(alignment, aligned_size);
        memset(ptr, 0, aligned_size);
        return ptr;
    }


    static inline void mem_free(void* ptr) {
        free(ptr);
    }


    static inline void coreneuron_abort() {
        abort();
    }

    // Allocate instance structure
    static void nrn_private_constructor_minipump(NrnThread* nt, Memb_list* ml, int type) {
        assert(!ml->instance);
        assert(!ml->global_variables);
        assert(ml->global_variables_size == 0);
        auto* const inst = new minipump_Instance{};
        assert(inst->global == &minipump_global);
        ml->instance = inst;
        ml->global_variables = inst->global;
        ml->global_variables_size = sizeof(minipump_Store);
    }

    // Deallocate the instance structure
    static void nrn_private_destructor_minipump(NrnThread* nt, Memb_list* ml, int type) {
        auto* const inst = static_cast<minipump_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &minipump_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(minipump_Store));
        delete inst;
        ml->instance = nullptr;
        ml->global_variables = nullptr;
        ml->global_variables_size = 0;
    }

    /** initialize mechanism instance variables */
    static inline void setup_instance(NrnThread* nt, Memb_list* ml) {
        auto* const inst = static_cast<minipump_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &minipump_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(minipump_Store));
        int pnodecount = ml->_nodecount_padded;
        Datum* indexes = ml->pdata;
        inst->X = ml->data+0*pnodecount;
        inst->Y = ml->data+1*pnodecount;
        inst->Z = ml->data+2*pnodecount;
        inst->DX = ml->data+3*pnodecount;
        inst->DY = ml->data+4*pnodecount;
        inst->DZ = ml->data+5*pnodecount;
        inst->v_unused = ml->data+6*pnodecount;
        inst->g_unused = ml->data+7*pnodecount;
    }



    static void nrn_alloc_minipump(double* data, Datum* indexes, int type) {
        // do nothing
    }


    void nrn_constructor_minipump(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<minipump_Instance*>(ml->instance);

        #endif
    }


    void nrn_destructor_minipump(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<minipump_Instance*>(ml->instance);

        #endif
    }


    struct functor_minipump_1 {
        NrnThread* nt;
        minipump_Instance* inst;
        int id;
        int pnodecount;
        double v;
        const Datum* indexes;
        double* data;
        ThreadDatum* thread;
        double kf0_, kb0_, old_X, old_Y;

        void initialize() {
            kf0_ = inst->global->kf;
            kb0_ = inst->global->kb;
            old_X = inst->X[id];
            old_Y = inst->Y[id];
        }

        functor_minipump_1(NrnThread* nt, minipump_Instance* inst, int id, int pnodecount, double v, const Datum* indexes, double* data, ThreadDatum* thread)
            : nt(nt), inst(inst), id(id), pnodecount(pnodecount), v(v), indexes(indexes), data(data), thread(thread)
        {}
        void operator()(const Eigen::Matrix<double, 3, 1>& nmodl_eigen_xm, Eigen::Matrix<double, 3, 1>& nmodl_eigen_fm, Eigen::Matrix<double, 3, 3>& nmodl_eigen_jm) const {
            const double* nmodl_eigen_x = nmodl_eigen_xm.data();
            double* nmodl_eigen_j = nmodl_eigen_jm.data();
            double* nmodl_eigen_f = nmodl_eigen_fm.data();
            nmodl_eigen_f[static_cast<int>(0)] = (nt->_dt * ( -nmodl_eigen_x[static_cast<int>(0)] * nmodl_eigen_x[static_cast<int>(1)] * kf0_ + nmodl_eigen_x[static_cast<int>(2)] * kb0_) + inst->global->volA * ( -nmodl_eigen_x[static_cast<int>(0)] + old_X)) / (nt->_dt * inst->global->volA);
            nmodl_eigen_j[static_cast<int>(0)] =  -nmodl_eigen_x[static_cast<int>(1)] * kf0_ / inst->global->volA - 1.0 / nt->_dt;
            nmodl_eigen_j[static_cast<int>(3)] =  -nmodl_eigen_x[static_cast<int>(0)] * kf0_ / inst->global->volA;
            nmodl_eigen_j[static_cast<int>(6)] = kb0_ / inst->global->volA;
            nmodl_eigen_f[static_cast<int>(1)] = (nt->_dt * ( -nmodl_eigen_x[static_cast<int>(0)] * nmodl_eigen_x[static_cast<int>(1)] * kf0_ + nmodl_eigen_x[static_cast<int>(2)] * kb0_) + inst->global->volB * ( -nmodl_eigen_x[static_cast<int>(1)] + old_Y)) / (nt->_dt * inst->global->volB);
            nmodl_eigen_j[static_cast<int>(1)] =  -nmodl_eigen_x[static_cast<int>(1)] * kf0_ / inst->global->volB;
            nmodl_eigen_j[static_cast<int>(4)] =  -nmodl_eigen_x[static_cast<int>(0)] * kf0_ / inst->global->volB - 1.0 / nt->_dt;
            nmodl_eigen_j[static_cast<int>(7)] = kb0_ / inst->global->volB;
            nmodl_eigen_f[static_cast<int>(2)] = ( -nmodl_eigen_x[static_cast<int>(1)] * inst->global->volB + 8.0 * inst->global->volB + inst->global->volC * (1.0 - nmodl_eigen_x[static_cast<int>(2)])) / inst->global->volC;
            nmodl_eigen_j[static_cast<int>(2)] = 0.0;
            nmodl_eigen_j[static_cast<int>(5)] =  -inst->global->volB / inst->global->volC;
            nmodl_eigen_j[static_cast<int>(8)] =  -1.0;
        }

        void finalize() {
        }
    };


    struct functor_minipump_0 {
        NrnThread* nt;
        minipump_Instance* inst;
        int id;
        int pnodecount;
        double v;
        const Datum* indexes;
        double* data;
        ThreadDatum* thread;
        double kf0_, kb0_, old_X, old_Y;

        void initialize() {
            ;
            kf0_ = inst->global->kf;
            kb0_ = inst->global->kb;
            old_X = inst->X[id];
            old_Y = inst->Y[id];
        }

        functor_minipump_0(NrnThread* nt, minipump_Instance* inst, int id, int pnodecount, double v, const Datum* indexes, double* data, ThreadDatum* thread)
            : nt(nt), inst(inst), id(id), pnodecount(pnodecount), v(v), indexes(indexes), data(data), thread(thread)
        {}
        void operator()(const Eigen::Matrix<double, 3, 1>& nmodl_eigen_xm, Eigen::Matrix<double, 3, 1>& nmodl_eigen_fm, Eigen::Matrix<double, 3, 3>& nmodl_eigen_jm) const {
            const double* nmodl_eigen_x = nmodl_eigen_xm.data();
            double* nmodl_eigen_j = nmodl_eigen_jm.data();
            double* nmodl_eigen_f = nmodl_eigen_fm.data();
            nmodl_eigen_f[static_cast<int>(0)] = (nt->_dt * ( -nmodl_eigen_x[static_cast<int>(0)] * nmodl_eigen_x[static_cast<int>(1)] * kf0_ + nmodl_eigen_x[static_cast<int>(2)] * kb0_) + inst->global->volA * ( -nmodl_eigen_x[static_cast<int>(0)] + old_X)) / (nt->_dt * inst->global->volA);
            nmodl_eigen_j[static_cast<int>(0)] =  -nmodl_eigen_x[static_cast<int>(1)] * kf0_ / inst->global->volA - 1.0 / nt->_dt;
            nmodl_eigen_j[static_cast<int>(3)] =  -nmodl_eigen_x[static_cast<int>(0)] * kf0_ / inst->global->volA;
            nmodl_eigen_j[static_cast<int>(6)] = kb0_ / inst->global->volA;
            nmodl_eigen_f[static_cast<int>(1)] = (nt->_dt * ( -nmodl_eigen_x[static_cast<int>(0)] * nmodl_eigen_x[static_cast<int>(1)] * kf0_ + nmodl_eigen_x[static_cast<int>(2)] * kb0_) + inst->global->volB * ( -nmodl_eigen_x[static_cast<int>(1)] + old_Y)) / (nt->_dt * inst->global->volB);
            nmodl_eigen_j[static_cast<int>(1)] =  -nmodl_eigen_x[static_cast<int>(1)] * kf0_ / inst->global->volB;
            nmodl_eigen_j[static_cast<int>(4)] =  -nmodl_eigen_x[static_cast<int>(0)] * kf0_ / inst->global->volB - 1.0 / nt->_dt;
            nmodl_eigen_j[static_cast<int>(7)] = kb0_ / inst->global->volB;
            nmodl_eigen_f[static_cast<int>(2)] = ( -nmodl_eigen_x[static_cast<int>(1)] * inst->global->volB + 8.0 * inst->global->volB + inst->global->volC * (1.0 - nmodl_eigen_x[static_cast<int>(2)])) / inst->global->volC;
            nmodl_eigen_j[static_cast<int>(2)] = 0.0;
            nmodl_eigen_j[static_cast<int>(5)] =  -inst->global->volB / inst->global->volC;
            nmodl_eigen_j[static_cast<int>(8)] =  -1.0;
        }

        void finalize() {
        }
    };


    /** initialize channel */
    void nrn_init_minipump(NrnThread* nt, Memb_list* ml, int type) {
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;

        setup_instance(nt, ml);
        auto* const inst = static_cast<minipump_Instance*>(ml->instance);

        if (_nrn_skip_initmodel == 0) {
            double _save_prev_dt = nt->_dt;
            nt->_dt = 1000000000;
            #pragma omp simd
            #pragma ivdep
            for (int id = 0; id < nodecount; id++) {
                int node_id = node_index[id];
                double v = voltage[node_id];
                #if NRN_PRCELLSTATE
                inst->v_unused[id] = v;
                #endif
                inst->X[id] = inst->global->X0;
                inst->Y[id] = inst->global->Y0;
                inst->Z[id] = inst->global->Z0;
                inst->X[id] = 40.0;
                inst->Y[id] = 8.0;
                inst->Z[id] = 1.0;
                if (inst->global->run_steady_state > 0.0) {
                                        
                    Eigen::Matrix<double, 3, 1> nmodl_eigen_xm;
                    double* nmodl_eigen_x = nmodl_eigen_xm.data();
                    nmodl_eigen_x[static_cast<int>(0)] = inst->X[id];
                    nmodl_eigen_x[static_cast<int>(1)] = inst->Y[id];
                    nmodl_eigen_x[static_cast<int>(2)] = inst->Z[id];
                    // call newton solver
                    functor_minipump_0 newton_functor(nt, inst, id, pnodecount, v, indexes, data, thread);
                    newton_functor.initialize();
                    int newton_iterations = nmodl::newton::newton_solver(nmodl_eigen_xm, newton_functor);
                    if (newton_iterations < 0) assert(false && "Newton solver did not converge!");
                    inst->X[id] = nmodl_eigen_x[static_cast<int>(0)];
                    inst->Y[id] = nmodl_eigen_x[static_cast<int>(1)];
                    inst->Z[id] = nmodl_eigen_x[static_cast<int>(2)];
                    newton_functor.initialize(); // TODO mimic calling F again.
                    newton_functor.finalize();


                }
            }
            nt->_dt = _save_prev_dt;
        }
    }


    /** update state */
    void nrn_state_minipump(NrnThread* nt, Memb_list* ml, int type) {
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<minipump_Instance*>(ml->instance);

        #pragma omp simd
        #pragma ivdep
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_index[id];
            double v = voltage[node_id];
            #if NRN_PRCELLSTATE
            inst->v_unused[id] = v;
            #endif
            
            Eigen::Matrix<double, 3, 1> nmodl_eigen_xm;
            double* nmodl_eigen_x = nmodl_eigen_xm.data();
            nmodl_eigen_x[static_cast<int>(0)] = inst->X[id];
            nmodl_eigen_x[static_cast<int>(1)] = inst->Y[id];
            nmodl_eigen_x[static_cast<int>(2)] = inst->Z[id];
            // call newton solver
            functor_minipump_1 newton_functor(nt, inst, id, pnodecount, v, indexes, data, thread);
            newton_functor.initialize();
            int newton_iterations = nmodl::newton::newton_solver(nmodl_eigen_xm, newton_functor);
            if (newton_iterations < 0) assert(false && "Newton solver did not converge!");
            inst->X[id] = nmodl_eigen_x[static_cast<int>(0)];
            inst->Y[id] = nmodl_eigen_x[static_cast<int>(1)];
            inst->Z[id] = nmodl_eigen_x[static_cast<int>(2)];
            newton_functor.initialize(); // TODO mimic calling F again.
            newton_functor.finalize();

        }
    }


    /** register channel with the simulator */
    void _minipump_reg() {

        int mech_type = nrn_get_mechtype("minipump");
        minipump_global.mech_type = mech_type;
        if (mech_type == -1) {
            return;
        }

        _nrn_layout_reg(mech_type, 0);
        register_mech(mechanism_info, nrn_alloc_minipump, nullptr, nullptr, nrn_state_minipump, nrn_init_minipump, nrn_private_constructor_minipump, nrn_private_destructor_minipump, first_pointer_var_index(), 1);

        hoc_register_prop_size(mech_type, float_variables_size(), int_variables_size());
        hoc_register_var(hoc_scalar_double, hoc_vector_double, NULL);
    }
}