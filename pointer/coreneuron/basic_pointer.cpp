/*********************************************************
Model Name      : basic_pointer
Filename        : basic_pointer.mod
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
        "basic_pointer",
        0,
        "x1_basic_pointer",
        "x2_basic_pointer",
        "ignore_basic_pointer",
        0,
        0,
        "p1_basic_pointer",
        "p2_basic_pointer",
        0
    };


    /** all global variables */
    struct basic_pointer_Store {
        int ca_type{};
        int reset{};
        int mech_type{};
    };
    static_assert(std::is_trivially_copy_constructible_v<basic_pointer_Store>);
    static_assert(std::is_trivially_move_constructible_v<basic_pointer_Store>);
    static_assert(std::is_trivially_copy_assignable_v<basic_pointer_Store>);
    static_assert(std::is_trivially_move_assignable_v<basic_pointer_Store>);
    static_assert(std::is_trivially_destructible_v<basic_pointer_Store>);
    static basic_pointer_Store basic_pointer_global;


    /** all mechanism instance variables and global variables */
    struct basic_pointer_Instance  {
        double* x1{};
        double* x2{};
        double* ignore{};
        double* ica{};
        double* v_unused{};
        const double* ion_ica{};
        double* p1{};
        double* p2{};
        basic_pointer_Store* global{&basic_pointer_global};
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
        return 1;
    }


    static inline int first_random_var_index() {
        return -1;
    }


    static inline int float_variables_size() {
        return 5;
    }


    static inline int int_variables_size() {
        return 3;
    }


    static inline int get_mech_type() {
        return basic_pointer_global.mech_type;
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
    static void nrn_private_constructor_basic_pointer(NrnThread* nt, Memb_list* ml, int type) {
        assert(!ml->instance);
        assert(!ml->global_variables);
        assert(ml->global_variables_size == 0);
        auto* const inst = new basic_pointer_Instance{};
        assert(inst->global == &basic_pointer_global);
        ml->instance = inst;
        ml->global_variables = inst->global;
        ml->global_variables_size = sizeof(basic_pointer_Store);
    }

    // Deallocate the instance structure
    static void nrn_private_destructor_basic_pointer(NrnThread* nt, Memb_list* ml, int type) {
        auto* const inst = static_cast<basic_pointer_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &basic_pointer_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(basic_pointer_Store));
        delete inst;
        ml->instance = nullptr;
        ml->global_variables = nullptr;
        ml->global_variables_size = 0;
    }

    /** initialize mechanism instance variables */
    static inline void setup_instance(NrnThread* nt, Memb_list* ml) {
        auto* const inst = static_cast<basic_pointer_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &basic_pointer_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(basic_pointer_Store));
        int pnodecount = ml->_nodecount_padded;
        Datum* indexes = ml->pdata;
        inst->x1 = ml->data+0*pnodecount;
        inst->x2 = ml->data+1*pnodecount;
        inst->ignore = ml->data+2*pnodecount;
        inst->ica = ml->data+3*pnodecount;
        inst->v_unused = ml->data+4*pnodecount;
        inst->ion_ica = nt->_data;
        inst->p1 = nt->_data;
        inst->p2 = nt->_data;
    }



    static void nrn_alloc_basic_pointer(double* data, Datum* indexes, int type) {
        // do nothing
    }


    void nrn_constructor_basic_pointer(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<basic_pointer_Instance*>(ml->instance);

        #endif
    }


    void nrn_destructor_basic_pointer(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<basic_pointer_Instance*>(ml->instance);

        #endif
    }


    inline double read_p1_basic_pointer(int id, int pnodecount, basic_pointer_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v);
    inline double read_p2_basic_pointer(int id, int pnodecount, basic_pointer_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v);


    inline double read_p1_basic_pointer(int id, int pnodecount, basic_pointer_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v) {
        double ret_read_p1 = 0.0;
        ret_read_p1 = inst->p1[indexes[1*pnodecount + id]];
        return ret_read_p1;
    }


    inline double read_p2_basic_pointer(int id, int pnodecount, basic_pointer_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v) {
        double ret_read_p2 = 0.0;
        ret_read_p2 = inst->p2[indexes[2*pnodecount + id]];
        return ret_read_p2;
    }


    /** initialize channel */
    void nrn_init_basic_pointer(NrnThread* nt, Memb_list* ml, int type) {
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;

        setup_instance(nt, ml);
        auto* const inst = static_cast<basic_pointer_Instance*>(ml->instance);

        if (_nrn_skip_initmodel == 0) {
            #pragma omp simd
            #pragma ivdep
            for (int id = 0; id < nodecount; id++) {
                int node_id = node_index[id];
                double v = voltage[node_id];
                #if NRN_PRCELLSTATE
                inst->v_unused[id] = v;
                #endif
                inst->ica[id] = inst->ion_ica[indexes[0*pnodecount + id]];
                inst->ignore[id] = inst->ica[id];
                inst->x1[id] = 0.0;
                inst->x2[id] = 0.0;
            }
        }
    }


    /** register channel with the simulator */
    void _basic_pointer_reg() {

        int mech_type = nrn_get_mechtype("basic_pointer");
        basic_pointer_global.mech_type = mech_type;
        if (mech_type == -1) {
            return;
        }

        _nrn_layout_reg(mech_type, 0);
        register_mech(mechanism_info, nrn_alloc_basic_pointer, nullptr, nullptr, nullptr, nrn_init_basic_pointer, nrn_private_constructor_basic_pointer, nrn_private_destructor_basic_pointer, first_pointer_var_index(), 1);
        basic_pointer_global.ca_type = nrn_get_mechtype("ca_ion");

        hoc_register_prop_size(mech_type, float_variables_size(), int_variables_size());
        hoc_register_dparam_semantics(mech_type, 0, "ca_ion");
        hoc_register_dparam_semantics(mech_type, 1, "pointer");
        hoc_register_dparam_semantics(mech_type, 2, "pointer");
        hoc_register_var(hoc_scalar_double, hoc_vector_double, NULL);
    }
}
