/*********************************************************
Model Name      : two_radii
Filename        : two_radii.mod
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
        "two_radii",
        0,
        "il_two_radii",
        "inv_two_radii",
        0,
        0,
        0
    };


    /** all global variables */
    struct two_radii_Store {
        int reset{};
        int mech_type{};
    };
    static_assert(std::is_trivially_copy_constructible_v<two_radii_Store>);
    static_assert(std::is_trivially_move_constructible_v<two_radii_Store>);
    static_assert(std::is_trivially_copy_assignable_v<two_radii_Store>);
    static_assert(std::is_trivially_move_assignable_v<two_radii_Store>);
    static_assert(std::is_trivially_destructible_v<two_radii_Store>);
    static two_radii_Store two_radii_global;


    /** all mechanism instance variables and global variables */
    struct two_radii_Instance  {
        double* il{};
        double* inv{};
        double* v_unused{};
        double* g_unused{};
        double* diam{};
        double* area{};
        two_radii_Store* global{&two_radii_global};
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
        return 4;
    }


    static inline int int_variables_size() {
        return 2;
    }


    static inline int get_mech_type() {
        return two_radii_global.mech_type;
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
    static void nrn_private_constructor_two_radii(NrnThread* nt, Memb_list* ml, int type) {
        assert(!ml->instance);
        assert(!ml->global_variables);
        assert(ml->global_variables_size == 0);
        auto* const inst = new two_radii_Instance{};
        assert(inst->global == &two_radii_global);
        ml->instance = inst;
        ml->global_variables = inst->global;
        ml->global_variables_size = sizeof(two_radii_Store);
    }

    // Deallocate the instance structure
    static void nrn_private_destructor_two_radii(NrnThread* nt, Memb_list* ml, int type) {
        auto* const inst = static_cast<two_radii_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &two_radii_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(two_radii_Store));
        delete inst;
        ml->instance = nullptr;
        ml->global_variables = nullptr;
        ml->global_variables_size = 0;
    }

    /** initialize mechanism instance variables */
    static inline void setup_instance(NrnThread* nt, Memb_list* ml) {
        auto* const inst = static_cast<two_radii_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &two_radii_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(two_radii_Store));
        int pnodecount = ml->_nodecount_padded;
        Datum* indexes = ml->pdata;
        inst->il = ml->data+0*pnodecount;
        inst->inv = ml->data+1*pnodecount;
        inst->v_unused = ml->data+2*pnodecount;
        inst->g_unused = ml->data+3*pnodecount;
        inst->diam = nt->_data;
        inst->area = nt->_data;
    }



    static void nrn_alloc_two_radii(double* data, Datum* indexes, int type) {
        // do nothing
    }


    void nrn_constructor_two_radii(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<two_radii_Instance*>(ml->instance);

        #endif
    }


    void nrn_destructor_two_radii(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<two_radii_Instance*>(ml->instance);

        #endif
    }


    inline static double square_diam_two_radii(int id, int pnodecount, two_radii_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v);
    inline static double square_area_two_radii(int id, int pnodecount, two_radii_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v);


    inline double square_diam_two_radii(int id, int pnodecount, two_radii_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v) {
        double ret_square_diam = 0.0;
        ret_square_diam = inst->diam[indexes[0*pnodecount + id]] * inst->diam[indexes[0*pnodecount + id]];
        return ret_square_diam;
    }


    inline double square_area_two_radii(int id, int pnodecount, two_radii_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v) {
        double ret_square_area = 0.0;
        ret_square_area = inst->area[indexes[1*pnodecount + id]] * inst->area[indexes[1*pnodecount + id]];
        return ret_square_area;
    }


    /** initialize channel */
    void nrn_init_two_radii(NrnThread* nt, Memb_list* ml, int type) {
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;

        setup_instance(nt, ml);
        auto* const inst = static_cast<two_radii_Instance*>(ml->instance);

        if (_nrn_skip_initmodel == 0) {
            #pragma omp simd
            #pragma ivdep
            for (int id = 0; id < nodecount; id++) {
                int node_id = node_index[id];
                double v = voltage[node_id];
                #if NRN_PRCELLSTATE
                inst->v_unused[id] = v;
                #endif
                inst->inv[id] = 1.0 / (square_diam_two_radii(id, pnodecount, inst, data, indexes, thread, nt, v) + inst->area[indexes[1*pnodecount + id]]);
            }
        }
    }


    inline double nrn_current_two_radii(int id, int pnodecount, two_radii_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v) {
        double current = 0.0;
        inst->il[id] = (square_diam_two_radii(id, pnodecount, inst, data, indexes, thread, nt, v) + inst->area[indexes[1*pnodecount + id]]) * 0.001 * (v - 20.0);
        current += inst->il[id];
        return current;
    }


    /** update current */
    void nrn_cur_two_radii(NrnThread* nt, Memb_list* ml, int type) {
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        double* vec_rhs = nt->_actual_rhs;
        double* vec_d = nt->_actual_d;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<two_radii_Instance*>(ml->instance);

        #pragma omp simd
        #pragma ivdep
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_index[id];
            double v = voltage[node_id];
            #if NRN_PRCELLSTATE
            inst->v_unused[id] = v;
            #endif
            double g = nrn_current_two_radii(id, pnodecount, inst, data, indexes, thread, nt, v+0.001);
            double rhs = nrn_current_two_radii(id, pnodecount, inst, data, indexes, thread, nt, v);
            g = (g-rhs)/0.001;
            #if NRN_PRCELLSTATE
            inst->g_unused[id] = g;
            #endif
            vec_rhs[node_id] -= rhs;
            vec_d[node_id] += g;
        }
    }


    /** update state */
    void nrn_state_two_radii(NrnThread* nt, Memb_list* ml, int type) {
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<two_radii_Instance*>(ml->instance);

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
    void _two_radii_reg() {

        int mech_type = nrn_get_mechtype("two_radii");
        two_radii_global.mech_type = mech_type;
        if (mech_type == -1) {
            return;
        }

        _nrn_layout_reg(mech_type, 0);
        register_mech(mechanism_info, nrn_alloc_two_radii, nrn_cur_two_radii, nullptr, nrn_state_two_radii, nrn_init_two_radii, nrn_private_constructor_two_radii, nrn_private_destructor_two_radii, first_pointer_var_index(), 1);

        hoc_register_prop_size(mech_type, float_variables_size(), int_variables_size());
        hoc_register_dparam_semantics(mech_type, 0, "diam");
        hoc_register_dparam_semantics(mech_type, 1, "area");
        hoc_register_var(hoc_scalar_double, hoc_vector_double, NULL);
    }
}
