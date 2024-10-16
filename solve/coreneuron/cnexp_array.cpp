/*********************************************************
Model Name      : cnexp_array
Filename        : cnexp_array.mod
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
        "cnexp_array",
        0,
        "z_cnexp_array[3]",
        0,
        "x_cnexp_array",
        "s_cnexp_array[2]",
        0,
        0
    };


    /** all global variables */
    struct cnexp_array_Store {
        double x0{};
        double s0{};
        int reset{};
        int mech_type{};
        int slist1[1]{3};
        int dlist1[1]{6};
    };
    static_assert(std::is_trivially_copy_constructible_v<cnexp_array_Store>);
    static_assert(std::is_trivially_move_constructible_v<cnexp_array_Store>);
    static_assert(std::is_trivially_copy_assignable_v<cnexp_array_Store>);
    static_assert(std::is_trivially_move_assignable_v<cnexp_array_Store>);
    static_assert(std::is_trivially_destructible_v<cnexp_array_Store>);
    cnexp_array_Store cnexp_array_global;


    /** all mechanism instance variables and global variables */
    struct cnexp_array_Instance  {
        double* z{};
        double* x{};
        double* s{};
        double* Dx{};
        double* Ds{};
        double* v_unused{};
        double* g_unused{};
        cnexp_array_Store* global{&cnexp_array_global};
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
        return 11;
    }


    static inline int int_variables_size() {
        return 0;
    }


    static inline int get_mech_type() {
        return cnexp_array_global.mech_type;
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
    static void nrn_private_constructor_cnexp_array(NrnThread* nt, Memb_list* ml, int type) {
        assert(!ml->instance);
        assert(!ml->global_variables);
        assert(ml->global_variables_size == 0);
        auto* const inst = new cnexp_array_Instance{};
        assert(inst->global == &cnexp_array_global);
        ml->instance = inst;
        ml->global_variables = inst->global;
        ml->global_variables_size = sizeof(cnexp_array_Store);
    }

    // Deallocate the instance structure
    static void nrn_private_destructor_cnexp_array(NrnThread* nt, Memb_list* ml, int type) {
        auto* const inst = static_cast<cnexp_array_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &cnexp_array_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(cnexp_array_Store));
        delete inst;
        ml->instance = nullptr;
        ml->global_variables = nullptr;
        ml->global_variables_size = 0;
    }

    /** initialize mechanism instance variables */
    static inline void setup_instance(NrnThread* nt, Memb_list* ml) {
        auto* const inst = static_cast<cnexp_array_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &cnexp_array_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(cnexp_array_Store));
        int pnodecount = ml->_nodecount_padded;
        Datum* indexes = ml->pdata;
        inst->z = ml->data+0*pnodecount;
        inst->x = ml->data+3*pnodecount;
        inst->s = ml->data+4*pnodecount;
        inst->Dx = ml->data+6*pnodecount;
        inst->Ds = ml->data+7*pnodecount;
        inst->v_unused = ml->data+9*pnodecount;
        inst->g_unused = ml->data+10*pnodecount;
    }



    static void nrn_alloc_cnexp_array(double* data, Datum* indexes, int type) {
        // do nothing
    }


    void nrn_constructor_cnexp_array(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<cnexp_array_Instance*>(ml->instance);

        #endif
    }


    void nrn_destructor_cnexp_array(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<cnexp_array_Instance*>(ml->instance);

        #endif
    }


    /** initialize channel */
    void nrn_init_cnexp_array(NrnThread* nt, Memb_list* ml, int type) {
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;

        setup_instance(nt, ml);
        auto* const inst = static_cast<cnexp_array_Instance*>(ml->instance);

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
                (inst->s+id*2)[0] = inst->global->s0;
                (inst->s+id*2)[1] = inst->global->s0;
                inst->x[id] = 42.0;
                (inst->s+id*2)[static_cast<int>(0)] = 0.1;
                (inst->s+id*2)[static_cast<int>(1)] =  -1.0;
                (inst->z+id*3)[static_cast<int>(0)] = 0.7;
                (inst->z+id*3)[static_cast<int>(1)] = 0.8;
                (inst->z+id*3)[static_cast<int>(2)] = 0.9;
            }
        }
    }


    /** update state */
    void nrn_state_cnexp_array(NrnThread* nt, Memb_list* ml, int type) {
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<cnexp_array_Instance*>(ml->instance);

        #pragma omp simd
        #pragma ivdep
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_index[id];
            double v = voltage[node_id];
            #if NRN_PRCELLSTATE
            inst->v_unused[id] = v;
            #endif
            inst->x[id] = inst->x[id] * exp(nt->_dt * ((inst->s+id*2)[static_cast<int>(0)] + (inst->s+id*2)[static_cast<int>(1)]) * (inst->z+id*3)[static_cast<int>(0)] * (inst->z+id*3)[static_cast<int>(1)] * (inst->z+id*3)[static_cast<int>(2)]);
        }
    }


    /** register channel with the simulator */
    void _cnexp_array_reg() {

        int mech_type = nrn_get_mechtype("cnexp_array");
        cnexp_array_global.mech_type = mech_type;
        if (mech_type == -1) {
            return;
        }

        _nrn_layout_reg(mech_type, 0);
        register_mech(mechanism_info, nrn_alloc_cnexp_array, nullptr, nullptr, nrn_state_cnexp_array, nrn_init_cnexp_array, nrn_private_constructor_cnexp_array, nrn_private_destructor_cnexp_array, first_pointer_var_index(), 1);

        hoc_register_prop_size(mech_type, float_variables_size(), int_variables_size());
        hoc_register_var(hoc_scalar_double, hoc_vector_double, NULL);
    }
}
