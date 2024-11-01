/*********************************************************
Model Name      : heat_eqn_scalar
Filename        : heat_eqn_scalar.mod
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
extern void _cvode_abstol(Symbol**, double*, int);
extern Node* nrn_alloc_node_;


namespace neuron {
    #ifndef NRN_PRCELLSTATE
    #define NRN_PRCELLSTATE 0
    #endif


    /** channel information */
    static const char *mechanism_info[] = {
        "7.7.0",
        "heat_eqn_scalar",
        0,
        "x_heat_eqn_scalar",
        0,
        "X_heat_eqn_scalar",
        0,
        0
    };


    /* NEURON global variables */
    static neuron::container::field_index _slist1[1], _dlist1[1];
    static Symbol** _atollist;
    static HocStateTolerance _hoc_state_tol[] = {
        {0, 0}
    };
    static int mech_type;
    static Prop* _extcall_prop;
    /* _prop_id kind of shadows _extcall_prop to allow validity checking. */
    static _nrn_non_owning_id_without_container _prop_id{};
    static _nrn_mechanism_std_vector<Datum> _extcall_thread;


    /** all global variables */
    struct heat_eqn_scalar_Store {
        double X0{0};
    };
    static_assert(std::is_trivially_copy_constructible_v<heat_eqn_scalar_Store>);
    static_assert(std::is_trivially_move_constructible_v<heat_eqn_scalar_Store>);
    static_assert(std::is_trivially_copy_assignable_v<heat_eqn_scalar_Store>);
    static_assert(std::is_trivially_move_assignable_v<heat_eqn_scalar_Store>);
    static_assert(std::is_trivially_destructible_v<heat_eqn_scalar_Store>);
    static heat_eqn_scalar_Store heat_eqn_scalar_global;
    auto X0_heat_eqn_scalar() -> std::decay<decltype(heat_eqn_scalar_global.X0)>::type  {
        return heat_eqn_scalar_global.X0;
    }

    static std::vector<double> _parameter_defaults = {
    };


    /** all mechanism instance variables and global variables */
    struct heat_eqn_scalar_Instance  {
        double* x{};
        double* X{};
        double* mu{};
        double* vol{};
        double* DX{};
        double* v_unused{};
        double* g_unused{};
        heat_eqn_scalar_Store* global{&heat_eqn_scalar_global};
    };


    struct heat_eqn_scalar_NodeData  {
        int const * nodeindices;
        double const * node_voltages;
        double * node_diagonal;
        double * node_rhs;
        int nodecount;
    };


    static heat_eqn_scalar_Instance make_instance_heat_eqn_scalar(_nrn_mechanism_cache_range* _lmc) {
        if(_lmc == nullptr) {
            return heat_eqn_scalar_Instance();
        }

        return heat_eqn_scalar_Instance {
            _lmc->template fpfield_ptr<0>(),
            _lmc->template fpfield_ptr<1>(),
            _lmc->template fpfield_ptr<2>(),
            _lmc->template fpfield_ptr<3>(),
            _lmc->template fpfield_ptr<4>(),
            _lmc->template fpfield_ptr<5>(),
            _lmc->template fpfield_ptr<6>()
        };
    }


    static heat_eqn_scalar_NodeData make_node_data_heat_eqn_scalar(NrnThread& nt, Memb_list& _ml_arg) {
        return heat_eqn_scalar_NodeData {
            _ml_arg.nodeindices,
            nt.node_voltage_storage(),
            nt.node_d_storage(),
            nt.node_rhs_storage(),
            _ml_arg.nodecount
        };
    }
    static heat_eqn_scalar_NodeData make_node_data_heat_eqn_scalar(Prop * _prop) {
        if(!_prop) {
            return heat_eqn_scalar_NodeData();
        }

        static std::vector<int> node_index{0};
        Node* _node = _nrn_mechanism_access_node(_prop);
        return heat_eqn_scalar_NodeData {
            node_index.data(),
            &_nrn_mechanism_access_voltage(_node),
            &_nrn_mechanism_access_d(_node),
            &_nrn_mechanism_access_rhs(_node),
            1
        };
    }

    static void nrn_destructor_heat_eqn_scalar(Prop* prop);


    static void nrn_alloc_heat_eqn_scalar(Prop* _prop) {
        Datum *_ppvar = nullptr;
        _ppvar = nrn_prop_datum_alloc(mech_type, 1, _prop);
        _nrn_mechanism_access_dparam(_prop) = _ppvar;
        _nrn_mechanism_cache_instance _lmc{_prop};
        size_t const _iml = 0;
        assert(_nrn_mechanism_get_num_vars(_prop) == 7);
        /*initialize range parameters*/
    }


    /* Mechanism procedures and functions */
    static void* _diffusion_space_X;
    static double _diffusion_coefficient_X(int _i, Memb_list* _ml_arg, size_t id, Datum* _ppvar, double* _pdvol, double* _pdfcdc, Datum* /* _thread */, NrnThread* nt, const _nrn_model_sorted_token& _sorted_token) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_heat_eqn_scalar(&_lmc);
        auto node_data = make_node_data_heat_eqn_scalar(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        *_pdvol= inst.vol[id];
        *_pdfcdc = 0.0;
        return inst.mu[id];
    }
    static void _apply_diffusion_function(ldifusfunc2_t _f, const _nrn_model_sorted_token& _sorted_token, NrnThread& _nt) {
        for(size_t _i = 0; _i < 1; ++_i) {
            (*_f)(mech_type, _diffusion_coefficient_X, &_diffusion_space_X, _i, /* x pos */ 1, /* Dx pos */ 4, _sorted_token, _nt);
        }
    }



    /* Functions related to CVODE codegen */
    static constexpr int ode_count_heat_eqn_scalar(int _type) {
        return 1;
    }


    static void ode_setup_tolerances_heat_eqn_scalar(Prop* _prop, int equation_index, neuron::container::data_handle<double>* _pv, neuron::container::data_handle<double>* _pvdot, double* _atol, int _type) {
        auto* _ppvar = _nrn_mechanism_access_dparam(_prop);
        _ppvar[0].literal_value<int>() = equation_index;
        for (int i = 0; i < ode_count_heat_eqn_scalar(0); i++) {
            _pv[i] = _nrn_mechanism_get_param_handle(_prop, _slist1[i]);
            _pvdot[i] = _nrn_mechanism_get_param_handle(_prop, _dlist1[i]);
            _cvode_abstol(_atollist, _atol, i);
        }
    }


    static int ode_update_nonstiff_heat_eqn_scalar(_nrn_mechanism_cache_range& _lmc, heat_eqn_scalar_Instance& inst, heat_eqn_scalar_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt) {
        auto v = node_data.node_voltages ? node_data.node_voltages[node_data.nodeindices[id]] : 0.0;
        double source0_;
        ;
        source0_ = 0.0;
        inst.DX[id] = ((source0_)) / (inst.vol[id]);
        return 0;
    }


    static int ode_update_stiff_heat_eqn_scalar(_nrn_mechanism_cache_range& _lmc, heat_eqn_scalar_Instance& inst, heat_eqn_scalar_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt) {
        auto v = node_data.node_voltages ? node_data.node_voltages[node_data.nodeindices[id]] : 0.0;
        double source0_;
        ;
        source0_ = 0.0;
        inst.DX[id] = inst.DX[id] / (1.0 - nt->_dt * (0.0));
        return 0;
    }


    static void ode_setup_nonstiff_heat_eqn_scalar(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_heat_eqn_scalar(&_lmc);
        auto node_data = make_node_data_heat_eqn_scalar(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
            auto* _ppvar = _ml_arg->pdata[id];
            auto v = node_data.node_voltages ? node_data.node_voltages[node_data.nodeindices[id]] : 0.0;
            ode_update_nonstiff_heat_eqn_scalar(_lmc, inst, node_data, id, _ppvar, _thread, nt);
        }
    }


    static void ode_setup_stiff_heat_eqn_scalar(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_heat_eqn_scalar(&_lmc);
        auto node_data = make_node_data_heat_eqn_scalar(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
            auto* _ppvar = _ml_arg->pdata[id];
            auto v = node_data.node_voltages ? node_data.node_voltages[node_data.nodeindices[id]] : 0.0;
            ode_update_stiff_heat_eqn_scalar(_lmc, inst, node_data, id, _ppvar, _thread, nt);
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


    struct functor_heat_eqn_scalar_0 {
        _nrn_mechanism_cache_range& _lmc;
        heat_eqn_scalar_Instance& inst;
        heat_eqn_scalar_NodeData& node_data;
        size_t id;
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        double source0_, old_X;

        void initialize() {
            ;
            source0_ = 0.0;
            old_X = inst.X[id];
        }

        functor_heat_eqn_scalar_0(_nrn_mechanism_cache_range& _lmc, heat_eqn_scalar_Instance& inst, heat_eqn_scalar_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt)
            : _lmc(_lmc), inst(inst), node_data(node_data), id(id), _ppvar(_ppvar), _thread(_thread), nt(nt)
        {}
        void operator()(const Eigen::Matrix<double, 1, 1>& nmodl_eigen_xm, Eigen::Matrix<double, 1, 1>& nmodl_eigen_dxm, Eigen::Matrix<double, 1, 1>& nmodl_eigen_fm, Eigen::Matrix<double, 1, 1>& nmodl_eigen_jm) const {
            const double* nmodl_eigen_x = nmodl_eigen_xm.data();
            double* nmodl_eigen_dx = nmodl_eigen_dxm.data();
            double* nmodl_eigen_j = nmodl_eigen_jm.data();
            double* nmodl_eigen_f = nmodl_eigen_fm.data();
            nmodl_eigen_dx[0] = std::max(1e-6, 0.02*std::fabs(nmodl_eigen_x[0]));
            nmodl_eigen_f[static_cast<int>(0)] =  -nmodl_eigen_x[static_cast<int>(0)] / nt->_dt + source0_ / inst.vol[id] + old_X / nt->_dt;
            nmodl_eigen_j[static_cast<int>(0)] =  -1.0 / nt->_dt;
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
        {"setdata_heat_eqn_scalar", _hoc_setdata},
        {nullptr, nullptr}
    };
    static NPyDirectMechFunc npy_direct_func_proc[] = {
        {nullptr, nullptr}
    };


    static void nrn_init_heat_eqn_scalar(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_heat_eqn_scalar(&_lmc);
        auto node_data = make_node_data_heat_eqn_scalar(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        #pragma omp simd
        #pragma ivdep
        for (int id = 0; id < nodecount; id++) {
            auto* _ppvar = _ml_arg->pdata[id];
            int node_id = node_data.nodeindices[id];
            inst.v_unused[id] = node_data.node_voltages[node_id];
            inst.X[id] = inst.global->X0;
            inst.mu[id] = 1.1;
            inst.vol[id] = 0.01;
            if (inst.x[id] < 0.5) {
                inst.X[id] = 1.0;
            } else {
                inst.X[id] = 0.0;
            }
        }
    }


    static void nrn_state_heat_eqn_scalar(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_heat_eqn_scalar(&_lmc);
        auto node_data = make_node_data_heat_eqn_scalar(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        #pragma omp simd
        #pragma ivdep
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_data.nodeindices[id];
            auto* _ppvar = _ml_arg->pdata[id];
            inst.v_unused[id] = node_data.node_voltages[node_id];
            
            Eigen::Matrix<double, 1, 1> nmodl_eigen_xm;
            double* nmodl_eigen_x = nmodl_eigen_xm.data();
            nmodl_eigen_x[static_cast<int>(0)] = inst.X[id];
            // call newton solver
            functor_heat_eqn_scalar_0 newton_functor(_lmc, inst, node_data, id, _ppvar, _thread, nt);
            newton_functor.initialize();
            int newton_iterations = nmodl::newton::newton_solver(nmodl_eigen_xm, newton_functor);
            if (newton_iterations < 0) assert(false && "Newton solver did not converge!");
            inst.X[id] = nmodl_eigen_x[static_cast<int>(0)];
            newton_functor.initialize(); // TODO mimic calling F again.
            newton_functor.finalize();

        }
    }


    static void nrn_jacob_heat_eqn_scalar(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_heat_eqn_scalar(&_lmc);
        auto node_data = make_node_data_heat_eqn_scalar(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        #pragma omp simd
        #pragma ivdep
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_data.nodeindices[id];
            node_data.node_diagonal[node_id] += inst.g_unused[id];
        }
    }
    static void nrn_destructor_heat_eqn_scalar(Prop* prop) {
        Datum* _ppvar = _nrn_mechanism_access_dparam(prop);
        _nrn_mechanism_cache_instance _lmc{prop};
        const size_t id = 0;
        auto inst = make_instance_heat_eqn_scalar(prop ? &_lmc : nullptr);
        auto node_data = make_node_data_heat_eqn_scalar(prop);

    }


    static void _initlists() {
        /* X */
        _slist1[0] = {1, 0};
        /* DX */
        _dlist1[0] = {4, 0};
    }


    extern "C" void _heat_eqn_scalar_reg() {
        _initlists();

        register_mech(mechanism_info, nrn_alloc_heat_eqn_scalar, nullptr, nrn_jacob_heat_eqn_scalar, nrn_state_heat_eqn_scalar, nrn_init_heat_eqn_scalar, -1, 1);

        mech_type = nrn_get_mechtype(mechanism_info[1]);
        hoc_register_parm_default(mech_type, &_parameter_defaults);
        _nrn_mechanism_register_data_fields(mech_type,
            _nrn_mechanism_field<double>{"x"} /* 0 */,
            _nrn_mechanism_field<double>{"X"} /* 1 */,
            _nrn_mechanism_field<double>{"mu"} /* 2 */,
            _nrn_mechanism_field<double>{"vol"} /* 3 */,
            _nrn_mechanism_field<double>{"DX"} /* 4 */,
            _nrn_mechanism_field<double>{"v_unused"} /* 5 */,
            _nrn_mechanism_field<double>{"g_unused"} /* 6 */,
            _nrn_mechanism_field<int>{"cvode_ieq", "cvodeieq"} /* 0 */
        );

        hoc_register_prop_size(mech_type, 7, 1);
        hoc_register_ldifus1(_apply_diffusion_function);
        hoc_register_var(hoc_scalar_double, hoc_vector_double, hoc_intfunc);
        hoc_register_npy_direct(mech_type, npy_direct_func_proc);
        hoc_register_dparam_semantics(mech_type, 0, "cvodeieq");
        hoc_register_cvode(mech_type, ode_count_heat_eqn_scalar, ode_setup_tolerances_heat_eqn_scalar, ode_setup_nonstiff_heat_eqn_scalar, ode_setup_stiff_heat_eqn_scalar);
        hoc_register_tolerance(mech_type, _hoc_state_tol, &_atollist);
    }
}
