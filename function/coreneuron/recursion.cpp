/*********************************************************
Model Name      : recursion
Filename        : recursion.mod
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
        "recursion",
        0,
        0,
        0,
        0
    };


    /** all global variables */
    struct recursion_Store {
        int reset{};
        int mech_type{};
    };
    static_assert(std::is_trivially_copy_constructible_v<recursion_Store>);
    static_assert(std::is_trivially_move_constructible_v<recursion_Store>);
    static_assert(std::is_trivially_copy_assignable_v<recursion_Store>);
    static_assert(std::is_trivially_move_assignable_v<recursion_Store>);
    static_assert(std::is_trivially_destructible_v<recursion_Store>);
    recursion_Store recursion_global;


    /** all mechanism instance variables and global variables */
    struct recursion_Instance  {
        double* v_unused{};
        recursion_Store* global{&recursion_global};
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


    static inline int float_variables_size() {
        return 1;
    }


    static inline int int_variables_size() {
        return 0;
    }


    static inline int get_mech_type() {
        return recursion_global.mech_type;
    }


    static inline Memb_list* get_memb_list(NrnThread* nt) {
        if (!nt->_ml_list) {
            return nullptr;
        }
        return nt->_ml_list[get_mech_type()];
    }


    static inline void* mem_alloc(size_t num, size_t size, size_t alignment = 16) {
        void* ptr;
        posix_memalign(&ptr, alignment, num*size);
        memset(ptr, 0, size);
        return ptr;
    }


    static inline void mem_free(void* ptr) {
        free(ptr);
    }


    static inline void coreneuron_abort() {
        abort();
    }

    // Allocate instance structure
    static void nrn_private_constructor_recursion(NrnThread* nt, Memb_list* ml, int type) {
        assert(!ml->instance);
        assert(!ml->global_variables);
        assert(ml->global_variables_size == 0);
        auto* const inst = new recursion_Instance{};
        assert(inst->global == &recursion_global);
        ml->instance = inst;
        ml->global_variables = inst->global;
        ml->global_variables_size = sizeof(recursion_Store);
    }

    // Deallocate the instance structure
    static void nrn_private_destructor_recursion(NrnThread* nt, Memb_list* ml, int type) {
        auto* const inst = static_cast<recursion_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &recursion_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(recursion_Store));
        delete inst;
        ml->instance = nullptr;
        ml->global_variables = nullptr;
        ml->global_variables_size = 0;
    }

    /** initialize mechanism instance variables */
    static inline void setup_instance(NrnThread* nt, Memb_list* ml) {
        auto* const inst = static_cast<recursion_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &recursion_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(recursion_Store));
        int pnodecount = ml->_nodecount_padded;
        Datum* indexes = ml->pdata;
        inst->v_unused = ml->data+0*pnodecount;
    }



    static void nrn_alloc_recursion(double* data, Datum* indexes, int type) {
        // do nothing
    }


    void nrn_constructor_recursion(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<recursion_Instance*>(ml->instance);

        #endif
    }


    void nrn_destructor_recursion(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<recursion_Instance*>(ml->instance);

        #endif
    }


    inline double fibonacci_recursion(int id, int pnodecount, recursion_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double n);


    inline double fibonacci_recursion(int id, int pnodecount, recursion_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double n) {
        double ret_fibonacci = 0.0;
        if (n == 0.0 || n == 1.0) {
            ret_fibonacci = 1.0;
        } else {
            ret_fibonacci = fibonacci_recursion(id, pnodecount, inst, data, indexes, thread, nt, v, n - 1.0) + fibonacci_recursion(id, pnodecount, inst, data, indexes, thread, nt, v, n - 2.0);
        }
        return ret_fibonacci;
    }


    /** initialize channel */
    void nrn_init_recursion(NrnThread* nt, Memb_list* ml, int type) {
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;

        setup_instance(nt, ml);
        auto* const inst = static_cast<recursion_Instance*>(ml->instance);

        if (_nrn_skip_initmodel == 0) {
            #pragma omp simd
            #pragma ivdep
            for (int id = 0; id < nodecount; id++) {
                int node_id = node_index[id];
                double v = voltage[node_id];
                #if NRN_PRCELLSTATE
                inst->v_unused[id] = v;
                #endif
            }
        }
    }


    /** register channel with the simulator */
    void _recursion_reg() {

        int mech_type = nrn_get_mechtype("recursion");
        recursion_global.mech_type = mech_type;
        if (mech_type == -1) {
            return;
        }

        register_mech(mechanism_info, nrn_alloc_recursion, nullptr, nullptr, nullptr, nrn_init_recursion, nrn_private_constructor_recursion, nrn_private_destructor_recursion, first_pointer_var_index(), 1);

        hoc_register_prop_size(mech_type, float_variables_size(), int_variables_size());
        hoc_register_var(hoc_scalar_double, hoc_vector_double, NULL);
    }
}
