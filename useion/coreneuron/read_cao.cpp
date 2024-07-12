/*********************************************************
Model Name      : read_cao
Filename        : read_cao.mod
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
        "read_cao",
        0,
        "x_read_cao",
        0,
        0,
        0
    };


    /** all global variables */
    struct read_cao_Store {
        int ca_type{};
        int reset{};
        int mech_type{};
    };
    static_assert(std::is_trivially_copy_constructible_v<read_cao_Store>);
    static_assert(std::is_trivially_move_constructible_v<read_cao_Store>);
    static_assert(std::is_trivially_copy_assignable_v<read_cao_Store>);
    static_assert(std::is_trivially_move_assignable_v<read_cao_Store>);
    static_assert(std::is_trivially_destructible_v<read_cao_Store>);
    read_cao_Store read_cao_global;


    /** all mechanism instance variables and global variables */
    struct read_cao_Instance  {
        double* x{};
        double* cao{};
        double* v_unused{};
        const double* ion_cao{};
        const double* ion_cai{};
        read_cao_Store* global{&read_cao_global};
    };


    /** connect global (scalar) variables to hoc -- */
    static DoubScal hoc_scalar_double[] = {
        {nullptr, nullptr}
    };


    /** connect global (array) variables to hoc -- */
    static DoubVec hoc_vector_double[] = {
        {nullptr, nullptr, 0}
    };


    static int first_pointer_var_index() {
        return -1;
    }


    static int first_random_var_index() {
        return -1;
    }


    static int float_variables_size() {
        return 3;
    }


    static int int_variables_size() {
        return 2;
    }


    static int get_mech_type() {
        return read_cao_global.mech_type;
    }


    static Memb_list* get_memb_list(NrnThread* nt) {
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
    static void nrn_private_constructor_read_cao(NrnThread* nt, Memb_list* ml, int type) {
        assert(!ml->instance);
        assert(!ml->global_variables);
        assert(ml->global_variables_size == 0);
        auto* const inst = new read_cao_Instance{};
        assert(inst->global == &read_cao_global);
        ml->instance = inst;
        ml->global_variables = inst->global;
        ml->global_variables_size = sizeof(read_cao_Store);
    }

    // Deallocate the instance structure
    static void nrn_private_destructor_read_cao(NrnThread* nt, Memb_list* ml, int type) {
        auto* const inst = static_cast<read_cao_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &read_cao_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(read_cao_Store));
        delete inst;
        ml->instance = nullptr;
        ml->global_variables = nullptr;
        ml->global_variables_size = 0;
    }

    /** initialize mechanism instance variables */
    static inline void setup_instance(NrnThread* nt, Memb_list* ml) {
        auto* const inst = static_cast<read_cao_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &read_cao_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(read_cao_Store));
        int pnodecount = ml->_nodecount_padded;
        Datum* indexes = ml->pdata;
        inst->x = ml->data+0*pnodecount;
        inst->cao = ml->data+1*pnodecount;
        inst->v_unused = ml->data+2*pnodecount;
        inst->ion_cao = nt->_data;
        inst->ion_cai = nt->_data;
    }



    static void nrn_alloc_read_cao(double* data, Datum* indexes, int type) {
        // do nothing
    }


    void nrn_constructor_read_cao(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<read_cao_Instance*>(ml->instance);

        #endif
    }


    void nrn_destructor_read_cao(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<read_cao_Instance*>(ml->instance);

        #endif
    }


    /** initialize channel */
    void nrn_init_read_cao(NrnThread* nt, Memb_list* ml, int type) {
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;

        setup_instance(nt, ml);
        auto* const inst = static_cast<read_cao_Instance*>(ml->instance);

        if (_nrn_skip_initmodel == 0) {
            #pragma omp simd
            #pragma ivdep
            for (int id = 0; id < nodecount; id++) {
                int node_id = node_index[id];
                double v = voltage[node_id];
                #if NRN_PRCELLSTATE
                inst->v_unused[id] = v;
                #endif
                inst->cao[id] = inst->ion_cao[indexes[0*pnodecount + id]];
                inst->x[id] = inst->cao[id];
            }
        }
    }


    /** register channel with the simulator */
    void _read_cao_reg() {

        int mech_type = nrn_get_mechtype("read_cao");
        read_cao_global.mech_type = mech_type;
        if (mech_type == -1) {
            return;
        }

        _nrn_layout_reg(mech_type, 0);
        register_mech(mechanism_info, nrn_alloc_read_cao, nullptr, nullptr, nullptr, nrn_init_read_cao, nrn_private_constructor_read_cao, nrn_private_destructor_read_cao, first_pointer_var_index(), 1);
        read_cao_global.ca_type = nrn_get_mechtype("ca_ion");

        hoc_register_prop_size(mech_type, float_variables_size(), int_variables_size());
        hoc_register_dparam_semantics(mech_type, 0, "ca_ion");
        hoc_register_dparam_semantics(mech_type, 1, "ca_ion");
        hoc_register_var(hoc_scalar_double, hoc_vector_double, NULL);
    }
}
