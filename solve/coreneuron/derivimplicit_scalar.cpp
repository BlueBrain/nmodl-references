/*********************************************************
Model Name      : derivimplicit_scalar
Filename        : derivimplicit_scalar.mod
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
        "derivimplicit_scalar",
        0,
        0,
        "x_derivimplicit_scalar",
        0,
        0
    };


    /** all global variables */
    struct derivimplicit_scalar_Store {
        double x0{};
        int reset{};
        int mech_type{};
        int slist1[1]{0};
        int dlist1[1]{1};
        int slist2[1]{0};
        ThreadDatum ext_call_thread[3]{};
    };
    static_assert(std::is_trivially_copy_constructible_v<derivimplicit_scalar_Store>);
    static_assert(std::is_trivially_move_constructible_v<derivimplicit_scalar_Store>);
    static_assert(std::is_trivially_copy_assignable_v<derivimplicit_scalar_Store>);
    static_assert(std::is_trivially_move_assignable_v<derivimplicit_scalar_Store>);
    static_assert(std::is_trivially_destructible_v<derivimplicit_scalar_Store>);
    derivimplicit_scalar_Store derivimplicit_scalar_global;


    /** all mechanism instance variables and global variables */
    struct derivimplicit_scalar_Instance  {
        double* x{};
        double* Dx{};
        double* v_unused{};
        double* g_unused{};
        derivimplicit_scalar_Store* global{&derivimplicit_scalar_global};
    };


    /** connect global (scalar) variables to hoc -- */
    static DoubScal hoc_scalar_double[] = {
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
        return 4;
    }


    static inline int int_variables_size() {
        return 0;
    }


    static inline int get_mech_type() {
        return derivimplicit_scalar_global.mech_type;
    }


    static inline Memb_list* get_memb_list(NrnThread* nt) {
        if (!nt->_ml_list) {
            return nullptr;
        }
        return nt->_ml_list[get_mech_type()];
    }


    static inline void* mem_alloc(size_t num, size_t size, size_t alignment = 32) {
        size_t aligned_size = (num*size + alignment - 1) / alignment) * alignment;
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
    static void nrn_private_constructor_derivimplicit_scalar(NrnThread* nt, Memb_list* ml, int type) {
        assert(!ml->instance);
        assert(!ml->global_variables);
        assert(ml->global_variables_size == 0);
        auto* const inst = new derivimplicit_scalar_Instance{};
        assert(inst->global == &derivimplicit_scalar_global);
        ml->instance = inst;
        ml->global_variables = inst->global;
        ml->global_variables_size = sizeof(derivimplicit_scalar_Store);
    }

    // Deallocate the instance structure
    static void nrn_private_destructor_derivimplicit_scalar(NrnThread* nt, Memb_list* ml, int type) {
        auto* const inst = static_cast<derivimplicit_scalar_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &derivimplicit_scalar_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(derivimplicit_scalar_Store));
        delete inst;
        ml->instance = nullptr;
        ml->global_variables = nullptr;
        ml->global_variables_size = 0;
    }

    /** initialize mechanism instance variables */
    static inline void setup_instance(NrnThread* nt, Memb_list* ml) {
        auto* const inst = static_cast<derivimplicit_scalar_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &derivimplicit_scalar_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(derivimplicit_scalar_Store));
        int pnodecount = ml->_nodecount_padded;
        Datum* indexes = ml->pdata;
        inst->x = ml->data+0*pnodecount;
        inst->Dx = ml->data+1*pnodecount;
        inst->v_unused = ml->data+2*pnodecount;
        inst->g_unused = ml->data+3*pnodecount;
    }



    static void nrn_alloc_derivimplicit_scalar(double* data, Datum* indexes, int type) {
        // do nothing
    }


    void nrn_constructor_derivimplicit_scalar(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<derivimplicit_scalar_Instance*>(ml->instance);

        #endif
    }


    void nrn_destructor_derivimplicit_scalar(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<derivimplicit_scalar_Instance*>(ml->instance);

        #endif
    }


    namespace {
        struct _newton_dX_derivimplicit_scalar {
            int operator()(int id, int pnodecount, double* data, Datum* indexes, ThreadDatum* thread, NrnThread* nt, Memb_list* ml, double v) const {
                auto* const inst = static_cast<derivimplicit_scalar_Instance*>(ml->instance);
                double* savstate1 = static_cast<double*>(thread[dith1()].pval);
                auto const& slist1 = inst->global->slist1;
                auto const& dlist1 = inst->global->dlist1;
                double* dlist2 = static_cast<double*>(thread[dith1()].pval) + (1*pnodecount);
                inst->Dx[id] =  -inst->x[id];
                int counter = -1;
                for (int i=0; i<1; i++) {
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

    int dX_derivimplicit_scalar(int id, int pnodecount, double* data, Datum* indexes, ThreadDatum* thread, NrnThread* nt, Memb_list* ml, double v) {
        auto* const inst = static_cast<derivimplicit_scalar_Instance*>(ml->instance);
        double* savstate1 = (double*) thread[dith1()].pval;
        auto const& slist1 = inst->global->slist1;
        auto& slist2 = inst->global->slist2;
        double* dlist2 = static_cast<double*>(thread[dith1()].pval) + (1*pnodecount);
        for (int i=0; i<1; i++) {
            savstate1[i*pnodecount+id] = data[slist1[i]*pnodecount+id];
        }
        int reset = nrn_newton_thread(static_cast<NewtonSpace*>(*newtonspace1(thread)), 1, slist2, _newton_dX_derivimplicit_scalar{}, dlist2, id, pnodecount, data, indexes, thread, nt, ml, v);
        return reset;
    }




    /** initialize channel */
    void nrn_init_derivimplicit_scalar(NrnThread* nt, Memb_list* ml, int type) {
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;

        setup_instance(nt, ml);
        auto* const inst = static_cast<derivimplicit_scalar_Instance*>(ml->instance);


        int& deriv_advance_flag = *deriv1_advance(thread);
        deriv_advance_flag = 0;
        auto ns = newtonspace1(thread);
        auto& th = thread[dith1()];
        if (*ns == nullptr) {
            int vec_size = 2*1*pnodecount*sizeof(double);
            double* vec = makevector(vec_size);
            th.pval = vec;
            *ns = nrn_cons_newtonspace(1, pnodecount);
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
                inst->x[id] = inst->global->x0;
                inst->x[id] = 42.0;
            }
        }
        deriv_advance_flag = 1;
    }


    /** update state */
    void nrn_state_derivimplicit_scalar(NrnThread* nt, Memb_list* ml, int type) {
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<derivimplicit_scalar_Instance*>(ml->instance);

        #pragma omp simd
        #pragma ivdep
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_index[id];
            double v = voltage[node_id];
            #if NRN_PRCELLSTATE
            inst->v_unused[id] = v;
            #endif
            dX_derivimplicit_scalar(id, pnodecount, data, indexes, thread, nt, ml, v);
        }
    }


    /** register channel with the simulator */
    void _derivimplicit_scalar_reg() {

        int mech_type = nrn_get_mechtype("derivimplicit_scalar");
        derivimplicit_scalar_global.mech_type = mech_type;
        if (mech_type == -1) {
            return;
        }

        _nrn_layout_reg(mech_type, 0);
        register_mech(mechanism_info, nrn_alloc_derivimplicit_scalar, nullptr, nullptr, nrn_state_derivimplicit_scalar, nrn_init_derivimplicit_scalar, nrn_private_constructor_derivimplicit_scalar, nrn_private_destructor_derivimplicit_scalar, first_pointer_var_index(), 4);

        thread_mem_init(derivimplicit_scalar_global.ext_call_thread);
        _nrn_thread_reg0(mech_type, thread_mem_cleanup);
        _nrn_thread_reg1(mech_type, thread_mem_init);
        hoc_register_prop_size(mech_type, float_variables_size(), int_variables_size());
        hoc_register_var(hoc_scalar_double, hoc_vector_double, NULL);
    }
}
