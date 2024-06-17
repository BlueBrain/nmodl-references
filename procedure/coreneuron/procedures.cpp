/*********************************************************
Model Name      : procedures
Filename        : procedures.mod
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
        "procedures",
        0,
        "x_procedures",
        0,
        0,
        0
    };


    /** all global variables */
    struct procedures_Store {
        int reset{};
        int mech_type{};
    };
    static_assert(std::is_trivially_copy_constructible_v<procedures_Store>);
    static_assert(std::is_trivially_move_constructible_v<procedures_Store>);
    static_assert(std::is_trivially_copy_assignable_v<procedures_Store>);
    static_assert(std::is_trivially_move_assignable_v<procedures_Store>);
    static_assert(std::is_trivially_destructible_v<procedures_Store>);
    procedures_Store procedures_global;


    /** all mechanism instance variables and global variables */
    struct procedures_Instance  {
        double* x{};
        double* v_unused{};
        procedures_Store* global{&procedures_global};
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
        return 2;
    }


    static inline int int_variables_size() {
        return 0;
    }


    static inline int get_mech_type() {
        return procedures_global.mech_type;
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
    static void nrn_private_constructor_procedures(NrnThread* nt, Memb_list* ml, int type) {
        assert(!ml->instance);
        assert(!ml->global_variables);
        assert(ml->global_variables_size == 0);
        auto* const inst = new procedures_Instance{};
        assert(inst->global == &procedures_global);
        ml->instance = inst;
        ml->global_variables = inst->global;
        ml->global_variables_size = sizeof(procedures_Store);
    }

    // Deallocate the instance structure
    static void nrn_private_destructor_procedures(NrnThread* nt, Memb_list* ml, int type) {
        auto* const inst = static_cast<procedures_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &procedures_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(procedures_Store));
        delete inst;
        ml->instance = nullptr;
        ml->global_variables = nullptr;
        ml->global_variables_size = 0;
    }

    /** initialize mechanism instance variables */
    static inline void setup_instance(NrnThread* nt, Memb_list* ml) {
        auto* const inst = static_cast<procedures_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &procedures_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(procedures_Store));
        int pnodecount = ml->_nodecount_padded;
        Datum* indexes = ml->pdata;
        inst->x = ml->data+0*pnodecount;
        inst->v_unused = ml->data+1*pnodecount;
    }



    static void nrn_alloc_procedures(double* data, Datum* indexes, int type) {
        // do nothing
    }


    void nrn_constructor_procedures(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<procedures_Instance*>(ml->instance);

        #endif
    }


    void nrn_destructor_procedures(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<procedures_Instance*>(ml->instance);

        #endif
    }


    inline double identity_procedures(int id, int pnodecount, procedures_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double arg_v);
    inline int set_x_42_procedures(int id, int pnodecount, procedures_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v);
    inline int set_x_a_procedures(int id, int pnodecount, procedures_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double a);
    inline int set_a_x_procedures(int id, int pnodecount, procedures_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v);
    inline int set_x_v_procedures(int id, int pnodecount, procedures_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v);
    inline int set_x_just_v_procedures(int id, int pnodecount, procedures_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v);
    inline int set_x_just_vv_procedures(int id, int pnodecount, procedures_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double arg_v);


    inline int set_x_42_procedures(int id, int pnodecount, procedures_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v) {
        int ret_set_x_42 = 0;
        set_x_a_procedures(id, pnodecount, inst, data, indexes, thread, nt, v, 42.0);
        return ret_set_x_42;
    }


    inline int set_x_a_procedures(int id, int pnodecount, procedures_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double a) {
        int ret_set_x_a = 0;
        inst->x[id] = a;
        return ret_set_x_a;
    }


    inline int set_a_x_procedures(int id, int pnodecount, procedures_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v) {
        int ret_set_a_x = 0;
        double a;
        a = inst->x[id];
        return ret_set_a_x;
    }


    inline int set_x_v_procedures(int id, int pnodecount, procedures_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v) {
        int ret_set_x_v = 0;
        inst->x[id] = v;
        return ret_set_x_v;
    }


    inline int set_x_just_v_procedures(int id, int pnodecount, procedures_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v) {
        int ret_set_x_just_v = 0;
        inst->x[id] = identity_procedures(id, pnodecount, inst, data, indexes, thread, nt, v, v);
        return ret_set_x_just_v;
    }


    inline int set_x_just_vv_procedures(int id, int pnodecount, procedures_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double arg_v) {
        int ret_set_x_just_vv = 0;
        inst->x[id] = identity_procedures(id, pnodecount, inst, data, indexes, thread, nt, v, arg_v);
        return ret_set_x_just_vv;
    }


    inline double identity_procedures(int id, int pnodecount, procedures_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double arg_v) {
        double ret_identity = 0.0;
        ret_identity = arg_v;
        return ret_identity;
    }


    /** initialize channel */
    void nrn_init_procedures(NrnThread* nt, Memb_list* ml, int type) {
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;

        setup_instance(nt, ml);
        auto* const inst = static_cast<procedures_Instance*>(ml->instance);

        if (_nrn_skip_initmodel == 0) {
            #pragma omp simd
            #pragma ivdep
            for (int id = 0; id < nodecount; id++) {
                int node_id = node_index[id];
                double v = voltage[node_id];
                #if NRN_PRCELLSTATE
                inst->v_unused[id] = v;
                #endif
                set_a_x_procedures(id, pnodecount, inst, data, indexes, thread, nt, v);
            }
        }
    }


    /** register channel with the simulator */
    void _procedures_reg() {

        int mech_type = nrn_get_mechtype("procedures");
        procedures_global.mech_type = mech_type;
        if (mech_type == -1) {
            return;
        }

        register_mech(mechanism_info, nrn_alloc_procedures, nullptr, nullptr, nullptr, nrn_init_procedures, nrn_private_constructor_procedures, nrn_private_destructor_procedures, first_pointer_var_index(), 1);

        hoc_register_prop_size(mech_type, float_variables_size(), int_variables_size());
        hoc_register_var(hoc_scalar_double, hoc_vector_double, NULL);
    }
}
