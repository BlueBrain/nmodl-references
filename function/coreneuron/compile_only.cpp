/*********************************************************
Model Name      : func_in_breakpoint
Filename        : compile_only.mod
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
        "func_in_breakpoint",
        0,
        "il_func_in_breakpoint",
        0,
        0,
        0
    };


    /** all global variables */
    struct func_in_breakpoint_Store {
        int reset{};
        int mech_type{};
        double c{1};
    };
    static_assert(std::is_trivially_copy_constructible_v<func_in_breakpoint_Store>);
    static_assert(std::is_trivially_move_constructible_v<func_in_breakpoint_Store>);
    static_assert(std::is_trivially_copy_assignable_v<func_in_breakpoint_Store>);
    static_assert(std::is_trivially_move_assignable_v<func_in_breakpoint_Store>);
    static_assert(std::is_trivially_destructible_v<func_in_breakpoint_Store>);
    static func_in_breakpoint_Store func_in_breakpoint_global;


    /** all mechanism instance variables and global variables */
    struct func_in_breakpoint_Instance  {
        double* il{};
        double* v_unused{};
        double* g_unused{};
        func_in_breakpoint_Store* global{&func_in_breakpoint_global};
    };


    /** connect global (scalar) variables to hoc -- */
    static DoubScal hoc_scalar_double[] = {
        {"c_func_in_breakpoint", &func_in_breakpoint_global.c},
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
        return 3;
    }


    static inline int int_variables_size() {
        return 0;
    }


    static inline int get_mech_type() {
        return func_in_breakpoint_global.mech_type;
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
    static void nrn_private_constructor_func_in_breakpoint(NrnThread* nt, Memb_list* ml, int type) {
        assert(!ml->instance);
        assert(!ml->global_variables);
        assert(ml->global_variables_size == 0);
        auto* const inst = new func_in_breakpoint_Instance{};
        assert(inst->global == &func_in_breakpoint_global);
        ml->instance = inst;
        ml->global_variables = inst->global;
        ml->global_variables_size = sizeof(func_in_breakpoint_Store);
    }

    // Deallocate the instance structure
    static void nrn_private_destructor_func_in_breakpoint(NrnThread* nt, Memb_list* ml, int type) {
        auto* const inst = static_cast<func_in_breakpoint_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &func_in_breakpoint_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(func_in_breakpoint_Store));
        delete inst;
        ml->instance = nullptr;
        ml->global_variables = nullptr;
        ml->global_variables_size = 0;
    }

    /** initialize mechanism instance variables */
    static inline void setup_instance(NrnThread* nt, Memb_list* ml) {
        auto* const inst = static_cast<func_in_breakpoint_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &func_in_breakpoint_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(func_in_breakpoint_Store));
        int pnodecount = ml->_nodecount_padded;
        Datum* indexes = ml->pdata;
        inst->il = ml->data+0*pnodecount;
        inst->v_unused = ml->data+1*pnodecount;
        inst->g_unused = ml->data+2*pnodecount;
    }



    static void nrn_alloc_func_in_breakpoint(double* data, Datum* indexes, int type) {
        // do nothing
    }


    void nrn_constructor_func_in_breakpoint(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<func_in_breakpoint_Instance*>(ml->instance);

        #endif
    }


    void nrn_destructor_func_in_breakpoint(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<func_in_breakpoint_Instance*>(ml->instance);

        #endif
    }


    inline static int func_func_in_breakpoint(int id, int pnodecount, func_in_breakpoint_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v);
    inline static int func_with_v_func_in_breakpoint(int id, int pnodecount, func_in_breakpoint_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double _lv);
    inline static int func_with_other_func_in_breakpoint(int id, int pnodecount, func_in_breakpoint_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double _lq);


    inline int func_func_in_breakpoint(int id, int pnodecount, func_in_breakpoint_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v) {
        int ret_func = 0;
        return ret_func;
    }


    inline int func_with_v_func_in_breakpoint(int id, int pnodecount, func_in_breakpoint_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double _lv) {
        int ret_func_with_v = 0;
        return ret_func_with_v;
    }


    inline int func_with_other_func_in_breakpoint(int id, int pnodecount, func_in_breakpoint_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double _lq) {
        int ret_func_with_other = 0;
        return ret_func_with_other;
    }


    /** initialize channel */
    void nrn_init_func_in_breakpoint(NrnThread* nt, Memb_list* ml, int type) {
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;

        setup_instance(nt, ml);
        auto* const inst = static_cast<func_in_breakpoint_Instance*>(ml->instance);

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


    inline double nrn_current_func_in_breakpoint(int id, int pnodecount, func_in_breakpoint_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v) {
        double current = 0.0;
        func_func_in_breakpoint(id, pnodecount, inst, data, indexes, thread, nt, v);
        func_with_v_func_in_breakpoint(id, pnodecount, inst, data, indexes, thread, nt, v, v);
        func_with_other_func_in_breakpoint(id, pnodecount, inst, data, indexes, thread, nt, v, inst->global->c);
        current += inst->il[id];
        return current;
    }


    /** update current */
    void nrn_cur_func_in_breakpoint(NrnThread* nt, Memb_list* ml, int type) {
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        double* vec_rhs = nt->_actual_rhs;
        double* vec_d = nt->_actual_d;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<func_in_breakpoint_Instance*>(ml->instance);

        #pragma omp simd
        #pragma ivdep
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_index[id];
            double v = voltage[node_id];
            #if NRN_PRCELLSTATE
            inst->v_unused[id] = v;
            #endif
            double g = nrn_current_func_in_breakpoint(id, pnodecount, inst, data, indexes, thread, nt, v+0.001);
            double rhs = nrn_current_func_in_breakpoint(id, pnodecount, inst, data, indexes, thread, nt, v);
            g = (g-rhs)/0.001;
            #if NRN_PRCELLSTATE
            inst->g_unused[id] = g;
            #endif
            vec_rhs[node_id] -= rhs;
            vec_d[node_id] += g;
        }
    }


    /** update state */
    void nrn_state_func_in_breakpoint(NrnThread* nt, Memb_list* ml, int type) {
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<func_in_breakpoint_Instance*>(ml->instance);

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


    /** register channel with the simulator */
    void _compile_only_reg() {

        int mech_type = nrn_get_mechtype("func_in_breakpoint");
        func_in_breakpoint_global.mech_type = mech_type;
        if (mech_type == -1) {
            return;
        }

        _nrn_layout_reg(mech_type, 0);
        register_mech(mechanism_info, nrn_alloc_func_in_breakpoint, nrn_cur_func_in_breakpoint, nullptr, nrn_state_func_in_breakpoint, nrn_init_func_in_breakpoint, nrn_private_constructor_func_in_breakpoint, nrn_private_destructor_func_in_breakpoint, first_pointer_var_index(), 1);

        hoc_register_prop_size(mech_type, float_variables_size(), int_variables_size());
        hoc_register_var(hoc_scalar_double, hoc_vector_double, NULL);
    }
}
