/*********************************************************
Model Name      : cacur
Filename        : cacur.mod
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
        "cacur",
        "del_cacur",
        "dur_cacur",
        "amp_cacur",
        0,
        0,
        0,
        0
    };


    /** all global variables */
    struct cacur_Store {
        int ca_type{};
        int reset{};
        int mech_type{};
    };
    static_assert(std::is_trivially_copy_constructible_v<cacur_Store>);
    static_assert(std::is_trivially_move_constructible_v<cacur_Store>);
    static_assert(std::is_trivially_copy_assignable_v<cacur_Store>);
    static_assert(std::is_trivially_move_assignable_v<cacur_Store>);
    static_assert(std::is_trivially_destructible_v<cacur_Store>);
    cacur_Store cacur_global;


    /** all mechanism instance variables and global variables */
    struct cacur_Instance  {
        const double* del{};
        const double* dur{};
        const double* amp{};
        double* ica{};
        double* v_unused{};
        double* g_unused{};
        double* ion_ica{};
        double* ion_dicadv{};
        cacur_Store* global{&cacur_global};
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
        return 6;
    }


    static inline int int_variables_size() {
        return 2;
    }


    static inline int get_mech_type() {
        return cacur_global.mech_type;
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
    static void nrn_private_constructor_cacur(NrnThread* nt, Memb_list* ml, int type) {
        assert(!ml->instance);
        assert(!ml->global_variables);
        assert(ml->global_variables_size == 0);
        auto* const inst = new cacur_Instance{};
        assert(inst->global == &cacur_global);
        ml->instance = inst;
        ml->global_variables = inst->global;
        ml->global_variables_size = sizeof(cacur_Store);
    }

    // Deallocate the instance structure
    static void nrn_private_destructor_cacur(NrnThread* nt, Memb_list* ml, int type) {
        auto* const inst = static_cast<cacur_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &cacur_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(cacur_Store));
        delete inst;
        ml->instance = nullptr;
        ml->global_variables = nullptr;
        ml->global_variables_size = 0;
    }

    /** initialize mechanism instance variables */
    static inline void setup_instance(NrnThread* nt, Memb_list* ml) {
        auto* const inst = static_cast<cacur_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &cacur_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(cacur_Store));
        int pnodecount = ml->_nodecount_padded;
        Datum* indexes = ml->pdata;
        inst->del = ml->data+0*pnodecount;
        inst->dur = ml->data+1*pnodecount;
        inst->amp = ml->data+2*pnodecount;
        inst->ica = ml->data+3*pnodecount;
        inst->v_unused = ml->data+4*pnodecount;
        inst->g_unused = ml->data+5*pnodecount;
        inst->ion_ica = nt->_data;
        inst->ion_dicadv = nt->_data;
    }



    static void nrn_alloc_cacur(double* data, Datum* indexes, int type) {
        // do nothing
    }


    void nrn_constructor_cacur(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<cacur_Instance*>(ml->instance);

        #endif
    }


    void nrn_destructor_cacur(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<cacur_Instance*>(ml->instance);

        #endif
    }


    /** initialize channel */
    void nrn_init_cacur(NrnThread* nt, Memb_list* ml, int type) {
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;

        setup_instance(nt, ml);
        auto* const inst = static_cast<cacur_Instance*>(ml->instance);

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


    inline double nrn_current_cacur(int id, int pnodecount, cacur_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v) {
        double current = 0.0;
        if (inst->amp[id]) {
            at_time(nt, inst->del[id]);
            at_time(nt, inst->del[id] + inst->dur[id]);
        }
        if (nt->_t > inst->del[id] && nt->_t < inst->del[id] + inst->dur[id]) {
            inst->ica[id] = inst->amp[id];
        } else {
            inst->ica[id] = 0.0;
        }
        current += inst->ica[id];
        return current;
    }


    /** update current */
    void nrn_cur_cacur(NrnThread* nt, Memb_list* ml, int type) {
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        double* vec_rhs = nt->_actual_rhs;
        double* vec_d = nt->_actual_d;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<cacur_Instance*>(ml->instance);

        #pragma omp simd
        #pragma ivdep
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_index[id];
            double v = voltage[node_id];
            #if NRN_PRCELLSTATE
            inst->v_unused[id] = v;
            #endif
            double g = nrn_current_cacur(id, pnodecount, inst, data, indexes, thread, nt, v+0.001);
            double dica = inst->ica[id];
            double rhs = nrn_current_cacur(id, pnodecount, inst, data, indexes, thread, nt, v);
            g = (g-rhs)/0.001;
            inst->ion_dicadv[indexes[1*pnodecount + id]] += (dica-inst->ica[id])/0.001;
            inst->ion_ica[indexes[0*pnodecount + id]] += inst->ica[id];
            #if NRN_PRCELLSTATE
            inst->g_unused[id] = g;
            #endif
            vec_rhs[node_id] -= rhs;
            vec_d[node_id] += g;
        }
    }


    /** update state */
    void nrn_state_cacur(NrnThread* nt, Memb_list* ml, int type) {
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<cacur_Instance*>(ml->instance);

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
    void _cacur_reg() {

        int mech_type = nrn_get_mechtype("cacur");
        cacur_global.mech_type = mech_type;
        if (mech_type == -1) {
            return;
        }

        _nrn_layout_reg(mech_type, 0);
        register_mech(mechanism_info, nrn_alloc_cacur, nrn_cur_cacur, nullptr, nrn_state_cacur, nrn_init_cacur, nrn_private_constructor_cacur, nrn_private_destructor_cacur, first_pointer_var_index(), 1);
        cacur_global.ca_type = nrn_get_mechtype("ca_ion");

        hoc_register_prop_size(mech_type, float_variables_size(), int_variables_size());
        hoc_register_dparam_semantics(mech_type, 0, "ca_ion");
        hoc_register_dparam_semantics(mech_type, 1, "ca_ion");
        hoc_register_var(hoc_scalar_double, hoc_vector_double, NULL);
    }
}
