/*********************************************************
Model Name      : point_function_table
Filename        : point_function_table.mod
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
        "point_function_table",
        0,
        0,
        0,
        0
    };


    /** all global variables */
    struct point_function_table_Store {
        int point_type{};
        int reset{};
        int mech_type{};
        void* _ptable_cnst1{};
        void* _ptable_cnst2{};
        void* _ptable_tau1{};
        void* _ptable_tau2{};
    };
    static_assert(std::is_trivially_copy_constructible_v<point_function_table_Store>);
    static_assert(std::is_trivially_move_constructible_v<point_function_table_Store>);
    static_assert(std::is_trivially_copy_assignable_v<point_function_table_Store>);
    static_assert(std::is_trivially_move_assignable_v<point_function_table_Store>);
    static_assert(std::is_trivially_destructible_v<point_function_table_Store>);
    static point_function_table_Store point_function_table_global;


    /** all mechanism instance variables and global variables */
    struct point_function_table_Instance  {
        double* v_unused{};
        const double* node_area{};
        const int* point_process{};
        point_function_table_Store* global{&point_function_table_global};
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
        return 2;
    }


    static inline int get_mech_type() {
        return point_function_table_global.mech_type;
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
    static void nrn_private_constructor_point_function_table(NrnThread* nt, Memb_list* ml, int type) {
        assert(!ml->instance);
        assert(!ml->global_variables);
        assert(ml->global_variables_size == 0);
        auto* const inst = new point_function_table_Instance{};
        assert(inst->global == &point_function_table_global);
        ml->instance = inst;
        ml->global_variables = inst->global;
        ml->global_variables_size = sizeof(point_function_table_Store);
    }

    // Deallocate the instance structure
    static void nrn_private_destructor_point_function_table(NrnThread* nt, Memb_list* ml, int type) {
        auto* const inst = static_cast<point_function_table_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &point_function_table_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(point_function_table_Store));
        delete inst;
        ml->instance = nullptr;
        ml->global_variables = nullptr;
        ml->global_variables_size = 0;
    }

    /** initialize mechanism instance variables */
    static inline void setup_instance(NrnThread* nt, Memb_list* ml) {
        auto* const inst = static_cast<point_function_table_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &point_function_table_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(point_function_table_Store));
        int pnodecount = ml->_nodecount_padded;
        Datum* indexes = ml->pdata;
        inst->v_unused = ml->data+0*pnodecount;
        inst->node_area = nt->_data;
        inst->point_process = ml->pdata;
    }



    static void nrn_alloc_point_function_table(double* data, Datum* indexes, int type) {
        // do nothing
    }


    void nrn_constructor_point_function_table(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<point_function_table_Instance*>(ml->instance);

        #endif
    }


    void nrn_destructor_point_function_table(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<point_function_table_Instance*>(ml->instance);

        #endif
    }


    inline static double use_tau2_point_function_table(int id, int pnodecount, point_function_table_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double _lv, double _lx);


    inline double use_tau2_point_function_table(int id, int pnodecount, point_function_table_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double _lv, double _lx) {
        double ret_use_tau2 = 0.0;
        ret_use_tau2 = tau2_point_function_table(id, pnodecount, inst, data, indexes, thread, nt, v, _lv, _lx);
        return ret_use_tau2;
    }
    double cnst1_point_function_table(int id, int pnodecount, point_function_table_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double v) {
        double _arg[1];
        _arg[0] = v;
        return hoc_func_table(inst->global->_ptable_cnst1, 1, _arg);
    }
    double table_cnst1_point_function_table(int id, int pnodecount, point_function_table_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v) {
        hoc_spec_table(&inst->global->_ptable_cnst1, 1);
        return 0.;
    }
    double cnst2_point_function_table(int id, int pnodecount, point_function_table_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double v, double x) {
        double _arg[2];
        _arg[0] = v;
        _arg[1] = x;
        return hoc_func_table(inst->global->_ptable_cnst2, 2, _arg);
    }
    double table_cnst2_point_function_table(int id, int pnodecount, point_function_table_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v) {
        hoc_spec_table(&inst->global->_ptable_cnst2, 2);
        return 0.;
    }
    double tau1_point_function_table(int id, int pnodecount, point_function_table_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double v) {
        double _arg[1];
        _arg[0] = v;
        return hoc_func_table(inst->global->_ptable_tau1, 1, _arg);
    }
    double table_tau1_point_function_table(int id, int pnodecount, point_function_table_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v) {
        hoc_spec_table(&inst->global->_ptable_tau1, 1);
        return 0.;
    }
    double tau2_point_function_table(int id, int pnodecount, point_function_table_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double v, double x) {
        double _arg[2];
        _arg[0] = v;
        _arg[1] = x;
        return hoc_func_table(inst->global->_ptable_tau2, 2, _arg);
    }
    double table_tau2_point_function_table(int id, int pnodecount, point_function_table_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v) {
        hoc_spec_table(&inst->global->_ptable_tau2, 2);
        return 0.;
    }


    /** initialize channel */
    void nrn_init_point_function_table(NrnThread* nt, Memb_list* ml, int type) {
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;

        setup_instance(nt, ml);
        auto* const inst = static_cast<point_function_table_Instance*>(ml->instance);

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
    void _point_function_table_reg() {

        int mech_type = nrn_get_mechtype("point_function_table");
        point_function_table_global.mech_type = mech_type;
        if (mech_type == -1) {
            return;
        }

        _nrn_layout_reg(mech_type, 0);
        point_register_mech(mechanism_info, nrn_alloc_point_function_table, nullptr, nullptr, nullptr, nrn_init_point_function_table, nrn_private_constructor_point_function_table, nrn_private_destructor_point_function_table, first_pointer_var_index(), nullptr, nullptr, 1);

        hoc_register_prop_size(mech_type, float_variables_size(), int_variables_size());
        hoc_register_dparam_semantics(mech_type, 0, "area");
        hoc_register_dparam_semantics(mech_type, 1, "pntproc");
        hoc_register_var(hoc_scalar_double, hoc_vector_double, NULL);
    }
}
