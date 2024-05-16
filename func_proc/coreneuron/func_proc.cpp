/*********************************************************
Model Name      : test_func_proc
Filename        : func_proc.mod
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
        "test_func_proc",
        0,
        "x_test_func_proc",
        0,
        0,
        0
    };


    /** all global variables */
    struct test_func_proc_Store {
        int reset{};
        int mech_type{};
    };
    static_assert(std::is_trivially_copy_constructible_v<test_func_proc_Store>);
    static_assert(std::is_trivially_move_constructible_v<test_func_proc_Store>);
    static_assert(std::is_trivially_copy_assignable_v<test_func_proc_Store>);
    static_assert(std::is_trivially_move_assignable_v<test_func_proc_Store>);
    static_assert(std::is_trivially_destructible_v<test_func_proc_Store>);
    test_func_proc_Store test_func_proc_global;


    /** all mechanism instance variables and global variables */
    struct test_func_proc_Instance  {
        double* x{};
        double* v_unused{};
        test_func_proc_Store* global{&test_func_proc_global};
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
        return test_func_proc_global.mech_type;
    }


    static inline Memb_list* get_memb_list(NrnThread* nt) {
        if (!nt->_ml_list) {
            return nullptr;
        }
        return nt->_ml_list[get_mech_type()];
    }


    static inline void* mem_alloc(size_t num, size_t size, size_t alignment = 32) {
        size_t aligned_size = std::ceil(static_cast<float>(num*size) / alignment) * alignment
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
    static void nrn_private_constructor_test_func_proc(NrnThread* nt, Memb_list* ml, int type) {
        assert(!ml->instance);
        assert(!ml->global_variables);
        assert(ml->global_variables_size == 0);
        auto* const inst = new test_func_proc_Instance{};
        assert(inst->global == &test_func_proc_global);
        ml->instance = inst;
        ml->global_variables = inst->global;
        ml->global_variables_size = sizeof(test_func_proc_Store);
    }

    // Deallocate the instance structure
    static void nrn_private_destructor_test_func_proc(NrnThread* nt, Memb_list* ml, int type) {
        auto* const inst = static_cast<test_func_proc_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &test_func_proc_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(test_func_proc_Store));
        delete inst;
        ml->instance = nullptr;
        ml->global_variables = nullptr;
        ml->global_variables_size = 0;
    }

    /** initialize mechanism instance variables */
    static inline void setup_instance(NrnThread* nt, Memb_list* ml) {
        auto* const inst = static_cast<test_func_proc_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &test_func_proc_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(test_func_proc_Store));
        int pnodecount = ml->_nodecount_padded;
        Datum* indexes = ml->pdata;
        inst->x = ml->data+0*pnodecount;
        inst->v_unused = ml->data+1*pnodecount;
    }



    static void nrn_alloc_test_func_proc(double* data, Datum* indexes, int type) {
        // do nothing
    }


    void nrn_constructor_test_func_proc(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<test_func_proc_Instance*>(ml->instance);

        #endif
    }


    void nrn_destructor_test_func_proc(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<test_func_proc_Instance*>(ml->instance);

        #endif
    }


    inline double x_plus_a_test_func_proc(int id, int pnodecount, test_func_proc_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double a);
    inline double v_plus_a_test_func_proc(int id, int pnodecount, test_func_proc_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double a);
    inline double just_v_test_func_proc(int id, int pnodecount, test_func_proc_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double arg_v);
    inline int set_x_42_test_func_proc(int id, int pnodecount, test_func_proc_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v);
    inline int set_x_a_test_func_proc(int id, int pnodecount, test_func_proc_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double a);
    inline int set_a_x_test_func_proc(int id, int pnodecount, test_func_proc_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v);
    inline int set_x_v_test_func_proc(int id, int pnodecount, test_func_proc_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v);
    inline int set_x_just_v_test_func_proc(int id, int pnodecount, test_func_proc_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v);
    inline int set_x_just_vv_test_func_proc(int id, int pnodecount, test_func_proc_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double arg_v);


    inline int set_x_42_test_func_proc(int id, int pnodecount, test_func_proc_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v) {
        int ret_set_x_42 = 0;
        set_x_a_test_func_proc(id, pnodecount, inst, data, indexes, thread, nt, v, 42.0);
        return ret_set_x_42;
    }


    inline int set_x_a_test_func_proc(int id, int pnodecount, test_func_proc_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double a) {
        int ret_set_x_a = 0;
        inst->x[id] = a;
        return ret_set_x_a;
    }


    inline int set_a_x_test_func_proc(int id, int pnodecount, test_func_proc_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v) {
        int ret_set_a_x = 0;
        double a;
        a = inst->x[id];
        return ret_set_a_x;
    }


    inline int set_x_v_test_func_proc(int id, int pnodecount, test_func_proc_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v) {
        int ret_set_x_v = 0;
        inst->x[id] = v;
        return ret_set_x_v;
    }


    inline int set_x_just_v_test_func_proc(int id, int pnodecount, test_func_proc_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v) {
        int ret_set_x_just_v = 0;
        inst->x[id] = just_v_test_func_proc(id, pnodecount, inst, data, indexes, thread, nt, v, v);
        return ret_set_x_just_v;
    }


    inline int set_x_just_vv_test_func_proc(int id, int pnodecount, test_func_proc_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double arg_v) {
        int ret_set_x_just_vv = 0;
        inst->x[id] = just_v_test_func_proc(id, pnodecount, inst, data, indexes, thread, nt, v, arg_v);
        return ret_set_x_just_vv;
    }


    inline double x_plus_a_test_func_proc(int id, int pnodecount, test_func_proc_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double a) {
        double ret_x_plus_a = 0.0;
        ret_x_plus_a = inst->x[id] + a;
        return ret_x_plus_a;
    }


    inline double v_plus_a_test_func_proc(int id, int pnodecount, test_func_proc_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double a) {
        double ret_v_plus_a = 0.0;
        ret_v_plus_a = v + a;
        return ret_v_plus_a;
    }


    inline double just_v_test_func_proc(int id, int pnodecount, test_func_proc_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double arg_v) {
        double ret_just_v = 0.0;
        ret_just_v = arg_v;
        return ret_just_v;
    }


    /** initialize channel */
    void nrn_init_test_func_proc(NrnThread* nt, Memb_list* ml, int type) {
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;

        setup_instance(nt, ml);
        auto* const inst = static_cast<test_func_proc_Instance*>(ml->instance);

        if (_nrn_skip_initmodel == 0) {
            #pragma omp simd
            #pragma ivdep
            for (int id = 0; id < nodecount; id++) {
                int node_id = node_index[id];
                double v = voltage[node_id];
                #if NRN_PRCELLSTATE
                inst->v_unused[id] = v;
                #endif
                set_a_x_test_func_proc(id, pnodecount, inst, data, indexes, thread, nt, v);
            }
        }
    }


    /** register channel with the simulator */
    void _func_proc_reg() {

        int mech_type = nrn_get_mechtype("test_func_proc");
        test_func_proc_global.mech_type = mech_type;
        if (mech_type == -1) {
            return;
        }

        _nrn_layout_reg(mech_type, 0);
        register_mech(mechanism_info, nrn_alloc_test_func_proc, nullptr, nullptr, nullptr, nrn_init_test_func_proc, nrn_private_constructor_test_func_proc, nrn_private_destructor_test_func_proc, first_pointer_var_index(), 1);

        hoc_register_prop_size(mech_type, float_variables_size(), int_variables_size());
        hoc_register_var(hoc_scalar_double, hoc_vector_double, NULL);
    }
}
