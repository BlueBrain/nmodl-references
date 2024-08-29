/*********************************************************
Model Name      : default_values
Filename        : default_values.mod
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
        "default_values",
        0,
        0,
        "X_default_values",
        "Y_default_values",
        "Z_default_values",
        "A_default_values[3]",
        "B_default_values[2]",
        0,
        0
    };


    /** all global variables */
    struct default_values_Store {
        double Y0{};
        int reset{};
        int mech_type{};
        double X0{2};
        double Z0{3};
        double A0{4};
        double B0{5};
    };
    static_assert(std::is_trivially_copy_constructible_v<default_values_Store>);
    static_assert(std::is_trivially_move_constructible_v<default_values_Store>);
    static_assert(std::is_trivially_copy_assignable_v<default_values_Store>);
    static_assert(std::is_trivially_move_assignable_v<default_values_Store>);
    static_assert(std::is_trivially_destructible_v<default_values_Store>);
    default_values_Store default_values_global;


    /** all mechanism instance variables and global variables */
    struct default_values_Instance  {
        double* X{};
        double* Y{};
        double* Z{};
        double* A{};
        double* B{};
        double* DX{};
        double* DY{};
        double* DZ{};
        double* DA{};
        double* DB{};
        double* v_unused{};
        default_values_Store* global{&default_values_global};
    };


    /** connect global (scalar) variables to hoc -- */
    static DoubScal hoc_scalar_double[] = {
        {"X0_default_values", &default_values_global.X0},
        {"Z0_default_values", &default_values_global.Z0},
        {"A0_default_values", &default_values_global.A0},
        {"B0_default_values", &default_values_global.B0},
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
        return 17;
    }


    static inline int int_variables_size() {
        return 0;
    }


    static inline int get_mech_type() {
        return default_values_global.mech_type;
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
    static void nrn_private_constructor_default_values(NrnThread* nt, Memb_list* ml, int type) {
        assert(!ml->instance);
        assert(!ml->global_variables);
        assert(ml->global_variables_size == 0);
        auto* const inst = new default_values_Instance{};
        assert(inst->global == &default_values_global);
        ml->instance = inst;
        ml->global_variables = inst->global;
        ml->global_variables_size = sizeof(default_values_Store);
    }

    // Deallocate the instance structure
    static void nrn_private_destructor_default_values(NrnThread* nt, Memb_list* ml, int type) {
        auto* const inst = static_cast<default_values_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &default_values_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(default_values_Store));
        delete inst;
        ml->instance = nullptr;
        ml->global_variables = nullptr;
        ml->global_variables_size = 0;
    }

    /** initialize mechanism instance variables */
    static inline void setup_instance(NrnThread* nt, Memb_list* ml) {
        auto* const inst = static_cast<default_values_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &default_values_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(default_values_Store));
        int pnodecount = ml->_nodecount_padded;
        Datum* indexes = ml->pdata;
        inst->X = ml->data+0*pnodecount;
        inst->Y = ml->data+1*pnodecount;
        inst->Z = ml->data+2*pnodecount;
        inst->A = ml->data+3*pnodecount;
        inst->B = ml->data+6*pnodecount;
        inst->DX = ml->data+8*pnodecount;
        inst->DY = ml->data+9*pnodecount;
        inst->DZ = ml->data+10*pnodecount;
        inst->DA = ml->data+11*pnodecount;
        inst->DB = ml->data+14*pnodecount;
        inst->v_unused = ml->data+16*pnodecount;
    }



    static void nrn_alloc_default_values(double* data, Datum* indexes, int type) {
        // do nothing
    }


    void nrn_constructor_default_values(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<default_values_Instance*>(ml->instance);

        #endif
    }


    void nrn_destructor_default_values(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<default_values_Instance*>(ml->instance);

        #endif
    }


    /** initialize channel */
    void nrn_init_default_values(NrnThread* nt, Memb_list* ml, int type) {
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;

        setup_instance(nt, ml);
        auto* const inst = static_cast<default_values_Instance*>(ml->instance);

        if (_nrn_skip_initmodel == 0) {
            #pragma omp simd
            #pragma ivdep
            for (int id = 0; id < nodecount; id++) {
                int node_id = node_index[id];
                double v = voltage[node_id];
                #if NRN_PRCELLSTATE
                inst->v_unused[id] = v;
                #endif
                inst->X[id] = inst->global->X0;
                inst->Y[id] = inst->global->Y0;
                inst->Z[id] = inst->global->Z0;
                (inst->A+id*3)[0] = inst->global->A0;
                (inst->A+id*3)[1] = inst->global->A0;
                (inst->A+id*3)[2] = inst->global->A0;
                (inst->B+id*2)[0] = inst->global->B0;
                (inst->B+id*2)[1] = inst->global->B0;
                inst->Z[id] = 7.0;
                (inst->B+id*2)[static_cast<int>(1)] = 8.0;
            }
        }
    }


    /** register channel with the simulator */
    void _default_values_reg() {

        int mech_type = nrn_get_mechtype("default_values");
        default_values_global.mech_type = mech_type;
        if (mech_type == -1) {
            return;
        }

        _nrn_layout_reg(mech_type, 0);
        register_mech(mechanism_info, nrn_alloc_default_values, nullptr, nullptr, nullptr, nrn_init_default_values, nrn_private_constructor_default_values, nrn_private_destructor_default_values, first_pointer_var_index(), 1);

        hoc_register_prop_size(mech_type, float_variables_size(), int_variables_size());
        hoc_register_var(hoc_scalar_double, hoc_vector_double, NULL);
    }
}
