/*********************************************************
Model Name      : write_cai
Filename        : write_cai.mod
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
        "write_cai",
        0,
        0,
        0,
        0
    };


    /** all global variables */
    struct write_cai_Store {
        int ca_type{};
        int reset{};
        int mech_type{};
    };
    static_assert(std::is_trivially_copy_constructible_v<write_cai_Store>);
    static_assert(std::is_trivially_move_constructible_v<write_cai_Store>);
    static_assert(std::is_trivially_copy_assignable_v<write_cai_Store>);
    static_assert(std::is_trivially_move_assignable_v<write_cai_Store>);
    static_assert(std::is_trivially_destructible_v<write_cai_Store>);
    static write_cai_Store write_cai_global;


    /** all mechanism instance variables and global variables */
    struct write_cai_Instance  {
        double* cai{};
        double* v_unused{};
        const double* ion_cao{};
        double* ion_cai{};
        double* ion_ca_erev{};
        const int* style_ca{};
        write_cai_Store* global{&write_cai_global};
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
        return 4;
    }


    static inline int get_mech_type() {
        return write_cai_global.mech_type;
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
    static void nrn_private_constructor_write_cai(NrnThread* nt, Memb_list* ml, int type) {
        assert(!ml->instance);
        assert(!ml->global_variables);
        assert(ml->global_variables_size == 0);
        auto* const inst = new write_cai_Instance{};
        assert(inst->global == &write_cai_global);
        ml->instance = inst;
        ml->global_variables = inst->global;
        ml->global_variables_size = sizeof(write_cai_Store);
    }

    // Deallocate the instance structure
    static void nrn_private_destructor_write_cai(NrnThread* nt, Memb_list* ml, int type) {
        auto* const inst = static_cast<write_cai_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &write_cai_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(write_cai_Store));
        delete inst;
        ml->instance = nullptr;
        ml->global_variables = nullptr;
        ml->global_variables_size = 0;
    }

    /** initialize mechanism instance variables */
    static inline void setup_instance(NrnThread* nt, Memb_list* ml) {
        auto* const inst = static_cast<write_cai_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &write_cai_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(write_cai_Store));
        int pnodecount = ml->_nodecount_padded;
        Datum* indexes = ml->pdata;
        inst->cai = ml->data+0*pnodecount;
        inst->v_unused = ml->data+1*pnodecount;
        inst->ion_cao = nt->_data;
        inst->ion_cai = nt->_data;
        inst->ion_ca_erev = nt->_data;
        inst->style_ca = ml->pdata;
    }



    static void nrn_alloc_write_cai(double* data, Datum* indexes, int type) {
        // do nothing
    }


    void nrn_constructor_write_cai(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<write_cai_Instance*>(ml->instance);

        #endif
    }


    void nrn_destructor_write_cai(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<write_cai_Instance*>(ml->instance);

        #endif
    }


    /** initialize channel */
    void nrn_init_write_cai(NrnThread* nt, Memb_list* ml, int type) {
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;

        setup_instance(nt, ml);
        auto* const inst = static_cast<write_cai_Instance*>(ml->instance);

        if (_nrn_skip_initmodel == 0) {
            #pragma omp simd
            #pragma ivdep
            for (int id = 0; id < nodecount; id++) {
                int node_id = node_index[id];
                double v = voltage[node_id];
                #if NRN_PRCELLSTATE
                inst->v_unused[id] = v;
                #endif
                inst->cai[id] = inst->ion_cai[indexes[1*pnodecount + id]];
                inst->cai[id] = 1124.0;
                inst->ion_cai[indexes[1*pnodecount + id]] = inst->cai[id];
                int ca_type = inst->global->ca_type;
                nrn_wrote_conc(ca_type, &(inst->ion_cai[indexes[1*pnodecount + id]]), 1, inst->style_ca[3], nrn_ion_global_map, celsius, nt->_ml_list[ca_type]->_nodecount_padded);
            }
        }
    }


    /** register channel with the simulator */
    void _write_cai_reg() {

        int mech_type = nrn_get_mechtype("write_cai");
        write_cai_global.mech_type = mech_type;
        if (mech_type == -1) {
            return;
        }

        _nrn_layout_reg(mech_type, 0);
        register_mech(mechanism_info, nrn_alloc_write_cai, nullptr, nullptr, nullptr, nrn_init_write_cai, nrn_private_constructor_write_cai, nrn_private_destructor_write_cai, first_pointer_var_index(), 1);
        write_cai_global.ca_type = nrn_get_mechtype("ca_ion");

        hoc_register_prop_size(mech_type, float_variables_size(), int_variables_size());
        hoc_register_dparam_semantics(mech_type, 0, "ca_ion");
        hoc_register_dparam_semantics(mech_type, 1, "ca_ion");
        hoc_register_dparam_semantics(mech_type, 2, "ca_ion");
        hoc_register_dparam_semantics(mech_type, 3, "#ca_ion");
        nrn_writes_conc(mech_type, 0);
        hoc_register_var(hoc_scalar_double, hoc_vector_double, NULL);
    }
}
