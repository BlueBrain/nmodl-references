/*********************************************************
Model Name      : scalar
Filename        : derivative.mod
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


namespace coreneuron {
    #ifndef NRN_PRCELLSTATE
    #define NRN_PRCELLSTATE 0
    #endif


    /** channel information */
    static const char *mechanism_info[] = {
        "7.7.0",
        "scalar",
        0,
        0,
        "var1_scalar",
        "var2_scalar",
        "var3_scalar",
        "var4_scalar",
        "var5_scalar",
        0,
        0
    };


    /** all global variables */
    struct scalar_Store {
        double var10{};
        double var20{};
        double var30{};
        double var40{};
        double var50{};
        int reset{};
        int mech_type{};
        double freq{10};
        double a{5};
        double v1{-1};
        double v2{5};
        double v3{15};
        double v4{0.8};
        double v5{0.3};
        double r{3};
        double k{0.2};
        double nmodl_alpha{1.2};
        double nmodl_beta{4.5};
        double nmodl_gamma{2.4};
        double nmodl_delta{7.5};
        int slist1[5]{0, 1, 2, 3, 4};
        int dlist1[5]{5, 6, 7, 8, 9};
        int slist2[5]{0, 1, 2, 3, 4};
        ThreadDatum ext_call_thread[3]{};
    };
    static_assert(std::is_trivially_copy_constructible_v<scalar_Store>);
    static_assert(std::is_trivially_move_constructible_v<scalar_Store>);
    static_assert(std::is_trivially_copy_assignable_v<scalar_Store>);
    static_assert(std::is_trivially_move_assignable_v<scalar_Store>);
    static_assert(std::is_trivially_destructible_v<scalar_Store>);
    scalar_Store scalar_global;


    /** all mechanism instance variables and global variables */
    struct scalar_Instance  {
        double* var1{};
        double* var2{};
        double* var3{};
        double* var4{};
        double* var5{};
        double* Dvar1{};
        double* Dvar2{};
        double* Dvar3{};
        double* Dvar4{};
        double* Dvar5{};
        double* v_unused{};
        double* g_unused{};
        scalar_Store* global{&scalar_global};
    };


    /** connect global (scalar) variables to hoc -- */
    static DoubScal hoc_scalar_double[] = {
        {"freq_scalar", &scalar_global.freq},
        {"a_scalar", &scalar_global.a},
        {"v1_scalar", &scalar_global.v1},
        {"v2_scalar", &scalar_global.v2},
        {"v3_scalar", &scalar_global.v3},
        {"v4_scalar", &scalar_global.v4},
        {"v5_scalar", &scalar_global.v5},
        {"r_scalar", &scalar_global.r},
        {"k_scalar", &scalar_global.k},
        {"nmodl_alpha_scalar", &scalar_global.nmodl_alpha},
        {"nmodl_beta_scalar", &scalar_global.nmodl_beta},
        {"nmodl_gamma_scalar", &scalar_global.nmodl_gamma},
        {"nmodl_delta_scalar", &scalar_global.nmodl_delta},
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


    /** thread specific helper routines for derivimplicit */

    static inline int* deriv1_advance(ThreadDatum* thread) {
        return &(thread[0].i);
    }

    static inline int dith1() {
        return 1;
    }

    static inline void** newtonspace1(ThreadDatum* thread) {
        return &(thread[2]._pvoid);
    }


    static inline int float_variables_size() {
        return 12;
    }


    static inline int int_variables_size() {
        return 0;
    }


    static inline int get_mech_type() {
        return scalar_global.mech_type;
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


    /** thread memory allocation callback */
    static void thread_mem_init(ThreadDatum* thread)  {
        thread[dith1()].pval = nullptr;
    }


    /** thread memory cleanup callback */
    static void thread_mem_cleanup(ThreadDatum* thread)  {
        free(thread[dith1()].pval);
        nrn_destroy_newtonspace(static_cast<NewtonSpace*>(*newtonspace1(thread)));
    }

    // Allocate instance structure
    static void nrn_private_constructor_scalar(NrnThread* nt, Memb_list* ml, int type) {
        assert(!ml->instance);
        assert(!ml->global_variables);
        assert(ml->global_variables_size == 0);
        auto* const inst = new scalar_Instance{};
        assert(inst->global == &scalar_global);
        ml->instance = inst;
        ml->global_variables = inst->global;
        ml->global_variables_size = sizeof(scalar_Store);
    }

    // Deallocate the instance structure
    static void nrn_private_destructor_scalar(NrnThread* nt, Memb_list* ml, int type) {
        auto* const inst = static_cast<scalar_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &scalar_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(scalar_Store));
        delete inst;
        ml->instance = nullptr;
        ml->global_variables = nullptr;
        ml->global_variables_size = 0;
    }

    /** initialize mechanism instance variables */
    static inline void setup_instance(NrnThread* nt, Memb_list* ml) {
        auto* const inst = static_cast<scalar_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &scalar_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(scalar_Store));
        int pnodecount = ml->_nodecount_padded;
        Datum* indexes = ml->pdata;
        inst->var1 = ml->data+0*pnodecount;
        inst->var2 = ml->data+1*pnodecount;
        inst->var3 = ml->data+2*pnodecount;
        inst->var4 = ml->data+3*pnodecount;
        inst->var5 = ml->data+4*pnodecount;
        inst->Dvar1 = ml->data+5*pnodecount;
        inst->Dvar2 = ml->data+6*pnodecount;
        inst->Dvar3 = ml->data+7*pnodecount;
        inst->Dvar4 = ml->data+8*pnodecount;
        inst->Dvar5 = ml->data+9*pnodecount;
        inst->v_unused = ml->data+10*pnodecount;
        inst->g_unused = ml->data+11*pnodecount;
    }



    static void nrn_alloc_scalar(double* data, Datum* indexes, int type) {
        // do nothing
    }


    void nrn_constructor_scalar(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<scalar_Instance*>(ml->instance);

        #endif
    }


    void nrn_destructor_scalar(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<scalar_Instance*>(ml->instance);

        #endif
    }


    namespace {
        struct _newton_equation_scalar {
            int operator()(int id, int pnodecount, double* data, Datum* indexes, ThreadDatum* thread, NrnThread* nt, Memb_list* ml, double v) const {
                auto* const inst = static_cast<scalar_Instance*>(ml->instance);
                double* savstate1 = static_cast<double*>(thread[dith1()].pval);
                auto const& slist1 = inst->global->slist1;
                auto const& dlist1 = inst->global->dlist1;
                double* dlist2 = static_cast<double*>(thread[dith1()].pval) + (5*pnodecount);
                inst->Dvar1[id] =  -sin(inst->global->freq * nt->_t);
                inst->Dvar2[id] =  -inst->var2[id] * inst->global->a;
                inst->Dvar3[id] = inst->global->r * inst->var3[id] * (1.0 - inst->var3[id] / inst->global->k);
                inst->Dvar4[id] = inst->global->nmodl_alpha * inst->var4[id] - inst->global->nmodl_beta * inst->var4[id] * inst->var5[id];
                inst->Dvar5[id] = inst->global->nmodl_delta * inst->var4[id] * inst->var5[id] - inst->global->nmodl_gamma * inst->var5[id];
                int counter = -1;
                for (int i=0; i<5; i++) {
                    if (*deriv1_advance(thread)) {
                        dlist2[(++counter)*pnodecount+id] = data[dlist1[i]*pnodecount+id]-(data[slist1[i]*pnodecount+id]-savstate1[i*pnodecount+id])/nt->_dt;
                    } else {
                        dlist2[(++counter)*pnodecount+id] = data[slist1[i]*pnodecount+id]-savstate1[i*pnodecount+id];
                    }
                }
                return 0;
            }
        };
    }

    int equation_scalar(int id, int pnodecount, double* data, Datum* indexes, ThreadDatum* thread, NrnThread* nt, Memb_list* ml, double v) {
        auto* const inst = static_cast<scalar_Instance*>(ml->instance);
        double* savstate1 = (double*) thread[dith1()].pval;
        auto const& slist1 = inst->global->slist1;
        auto& slist2 = inst->global->slist2;
        double* dlist2 = static_cast<double*>(thread[dith1()].pval) + (5*pnodecount);
        for (int i=0; i<5; i++) {
            savstate1[i*pnodecount+id] = data[slist1[i]*pnodecount+id];
        }
        int reset = nrn_newton_thread(static_cast<NewtonSpace*>(*newtonspace1(thread)), 5, slist2, _newton_equation_scalar{}, dlist2, id, pnodecount, data, indexes, thread, nt, ml, v);
        return reset;
    }




    /** initialize channel */
    void nrn_init_scalar(NrnThread* nt, Memb_list* ml, int type) {
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;

        setup_instance(nt, ml);
        auto* const inst = static_cast<scalar_Instance*>(ml->instance);


        int& deriv_advance_flag = *deriv1_advance(thread);
        deriv_advance_flag = 0;
        auto ns = newtonspace1(thread);
        auto& th = thread[dith1()];
        if (*ns == nullptr) {
            int vec_size = 2*5*pnodecount*sizeof(double);
            double* vec = makevector(vec_size);
            th.pval = vec;
            *ns = nrn_cons_newtonspace(5, pnodecount);
        }
        if (_nrn_skip_initmodel == 0) {
            #pragma omp simd
            #pragma ivdep
            for (int id = 0; id < nodecount; id++) {
                int node_id = node_index[id];
                double v = voltage[node_id];
                #if NRN_PRCELLSTATE
                inst->v_unused[id] = v;
                #endif
                inst->var1[id] = inst->global->var10;
                inst->var2[id] = inst->global->var20;
                inst->var3[id] = inst->global->var30;
                inst->var4[id] = inst->global->var40;
                inst->var5[id] = inst->global->var50;
                inst->var1[id] = inst->global->v1;
                inst->var2[id] = inst->global->v2;
                inst->var3[id] = inst->global->v3;
                inst->var4[id] = inst->global->v4;
                inst->var5[id] = inst->global->v5;
            }
        }
        deriv_advance_flag = 1;
    }


    /** update state */
    void nrn_state_scalar(NrnThread* nt, Memb_list* ml, int type) {
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<scalar_Instance*>(ml->instance);

        #pragma omp simd
        #pragma ivdep
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_index[id];
            double v = voltage[node_id];
            #if NRN_PRCELLSTATE
            inst->v_unused[id] = v;
            #endif
            equation_scalar(id, pnodecount, data, indexes, thread, nt, ml, v);
        }
    }


    /** register channel with the simulator */
    void _derivative_reg() {

        int mech_type = nrn_get_mechtype("scalar");
        scalar_global.mech_type = mech_type;
        if (mech_type == -1) {
            return;
        }

        _nrn_layout_reg(mech_type, 0);
        register_mech(mechanism_info, nrn_alloc_scalar, nullptr, nullptr, nrn_state_scalar, nrn_init_scalar, nrn_private_constructor_scalar, nrn_private_destructor_scalar, first_pointer_var_index(), 4);

        thread_mem_init(scalar_global.ext_call_thread);
        _nrn_thread_reg0(mech_type, thread_mem_cleanup);
        _nrn_thread_reg1(mech_type, thread_mem_init);
        hoc_register_prop_size(mech_type, float_variables_size(), int_variables_size());
        hoc_register_var(hoc_scalar_double, hoc_vector_double, NULL);
    }
}
