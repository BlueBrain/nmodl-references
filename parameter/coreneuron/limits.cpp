/*********************************************************
Model Name      : limits_mod
Filename        : limits.mod
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
        "limits_mod",
        0,
        0,
        0,
        0
    };


    /** all global variables */
    struct limits_mod_Store {
        int reset{};
        int mech_type{};
        double a{24};
        double b{24};
        double c{24};
        double x{24};
        double y{24};
        double z{24};
    };
    static_assert(std::is_trivially_copy_constructible_v<limits_mod_Store>);
    static_assert(std::is_trivially_move_constructible_v<limits_mod_Store>);
    static_assert(std::is_trivially_copy_assignable_v<limits_mod_Store>);
    static_assert(std::is_trivially_move_assignable_v<limits_mod_Store>);
    static_assert(std::is_trivially_destructible_v<limits_mod_Store>);
    limits_mod_Store limits_mod_global;


    /** all mechanism instance variables and global variables */
    struct limits_mod_Instance  {
        double* v_unused{};
        limits_mod_Store* global{&limits_mod_global};
    };


    /** connect global (scalar) variables to hoc -- */
    static DoubScal hoc_scalar_double[] = {
        {"a_limits_mod", &limits_mod_global.a},
        {"b_limits_mod", &limits_mod_global.b},
        {"c_limits_mod", &limits_mod_global.c},
        {"x_limits_mod", &limits_mod_global.x},
        {"y_limits_mod", &limits_mod_global.y},
        {"z_limits_mod", &limits_mod_global.z},
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
        return limits_mod_global.mech_type;
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
    static void nrn_private_constructor_limits_mod(NrnThread* nt, Memb_list* ml, int type) {
        assert(!ml->instance);
        assert(!ml->global_variables);
        assert(ml->global_variables_size == 0);
        auto* const inst = new limits_mod_Instance{};
        assert(inst->global == &limits_mod_global);
        ml->instance = inst;
        ml->global_variables = inst->global;
        ml->global_variables_size = sizeof(limits_mod_Store);
    }

    // Deallocate the instance structure
    static void nrn_private_destructor_limits_mod(NrnThread* nt, Memb_list* ml, int type) {
        auto* const inst = static_cast<limits_mod_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &limits_mod_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(limits_mod_Store));
        delete inst;
        ml->instance = nullptr;
        ml->global_variables = nullptr;
        ml->global_variables_size = 0;
    }

    /** initialize mechanism instance variables */
    static inline void setup_instance(NrnThread* nt, Memb_list* ml) {
        auto* const inst = static_cast<limits_mod_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &limits_mod_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(limits_mod_Store));
        int pnodecount = ml->_nodecount_padded;
        Datum* indexes = ml->pdata;
        inst->v_unused = ml->data+0*pnodecount;
    }



    static void nrn_alloc_limits_mod(double* data, Datum* indexes, int type) {
        // do nothing
    }


    void nrn_constructor_limits_mod(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<limits_mod_Instance*>(ml->instance);

        #endif
    }


    void nrn_destructor_limits_mod(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<limits_mod_Instance*>(ml->instance);

        #endif
    }


    /** initialize channel */
    void nrn_init_limits_mod(NrnThread* nt, Memb_list* ml, int type) {
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;

        setup_instance(nt, ml);
        auto* const inst = static_cast<limits_mod_Instance*>(ml->instance);

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
    void _limits_reg() {

        int mech_type = nrn_get_mechtype("limits_mod");
        limits_mod_global.mech_type = mech_type;
        if (mech_type == -1) {
            return;
        }

        _nrn_layout_reg(mech_type, 0);
        register_mech(mechanism_info, nrn_alloc_limits_mod, nullptr, nullptr, nullptr, nrn_init_limits_mod, nrn_private_constructor_limits_mod, nrn_private_destructor_limits_mod, first_pointer_var_index(), 1);

        hoc_register_prop_size(mech_type, float_variables_size(), int_variables_size());
        hoc_register_var(hoc_scalar_double, hoc_vector_double, NULL);
    }
}
