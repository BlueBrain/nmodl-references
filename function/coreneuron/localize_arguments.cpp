/*********************************************************
Model Name      : localize_arguments
Filename        : localize_arguments.mod
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
        "localize_arguments",
        0,
        "x_localize_arguments",
        0,
        "s_localize_arguments",
        0,
        0
    };


    /** all global variables */
    struct localize_arguments_Store {
        int na_type{};
        double s0{};
        int reset{};
        int mech_type{};
        double g{0};
        double p{42};
    };
    static_assert(std::is_trivially_copy_constructible_v<localize_arguments_Store>);
    static_assert(std::is_trivially_move_constructible_v<localize_arguments_Store>);
    static_assert(std::is_trivially_copy_assignable_v<localize_arguments_Store>);
    static_assert(std::is_trivially_move_assignable_v<localize_arguments_Store>);
    static_assert(std::is_trivially_destructible_v<localize_arguments_Store>);
    static localize_arguments_Store localize_arguments_global;


    /** all mechanism instance variables and global variables */
    struct localize_arguments_Instance  {
        double* x{};
        double* s{};
        double* ina{};
        double* nai{};
        double* Ds{};
        double* v_unused{};
        const double* ion_ina{};
        const double* ion_nai{};
        const double* ion_nao{};
        localize_arguments_Store* global{&localize_arguments_global};
    };


    /** connect global (scalar) variables to hoc -- */
    static DoubScal hoc_scalar_double[] = {
        {"g_localize_arguments", &localize_arguments_global.g},
        {"p_localize_arguments", &localize_arguments_global.p},
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
        return 3;
    }


    static inline int get_mech_type() {
        return localize_arguments_global.mech_type;
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
    static void nrn_private_constructor_localize_arguments(NrnThread* nt, Memb_list* ml, int type) {
        assert(!ml->instance);
        assert(!ml->global_variables);
        assert(ml->global_variables_size == 0);
        auto* const inst = new localize_arguments_Instance{};
        assert(inst->global == &localize_arguments_global);
        ml->instance = inst;
        ml->global_variables = inst->global;
        ml->global_variables_size = sizeof(localize_arguments_Store);
    }

    // Deallocate the instance structure
    static void nrn_private_destructor_localize_arguments(NrnThread* nt, Memb_list* ml, int type) {
        auto* const inst = static_cast<localize_arguments_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &localize_arguments_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(localize_arguments_Store));
        delete inst;
        ml->instance = nullptr;
        ml->global_variables = nullptr;
        ml->global_variables_size = 0;
    }

    /** initialize mechanism instance variables */
    static inline void setup_instance(NrnThread* nt, Memb_list* ml) {
        auto* const inst = static_cast<localize_arguments_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &localize_arguments_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(localize_arguments_Store));
        int pnodecount = ml->_nodecount_padded;
        Datum* indexes = ml->pdata;
        inst->x = ml->data+0*pnodecount;
        inst->s = ml->data+1*pnodecount;
        inst->ina = ml->data+2*pnodecount;
        inst->nai = ml->data+3*pnodecount;
        inst->Ds = ml->data+4*pnodecount;
        inst->v_unused = ml->data+5*pnodecount;
        inst->ion_ina = nt->_data;
        inst->ion_nai = nt->_data;
        inst->ion_nao = nt->_data;
    }



    static void nrn_alloc_localize_arguments(double* data, Datum* indexes, int type) {
        // do nothing
    }


    void nrn_constructor_localize_arguments(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<localize_arguments_Instance*>(ml->instance);

        #endif
    }


    void nrn_destructor_localize_arguments(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<localize_arguments_Instance*>(ml->instance);

        #endif
    }


    inline double id_v_localize_arguments(int id, int pnodecount, localize_arguments_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double _lv);
    inline double id_nai_localize_arguments(int id, int pnodecount, localize_arguments_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double _lnai);
    inline double id_ina_localize_arguments(int id, int pnodecount, localize_arguments_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double _lina);
    inline double id_x_localize_arguments(int id, int pnodecount, localize_arguments_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double _lx);
    inline double id_g_localize_arguments(int id, int pnodecount, localize_arguments_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double _lg);
    inline double id_s_localize_arguments(int id, int pnodecount, localize_arguments_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double _ls);
    inline double id_p_localize_arguments(int id, int pnodecount, localize_arguments_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double _lp);


    inline double id_v_localize_arguments(int id, int pnodecount, localize_arguments_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double _lv) {
        double ret_id_v = 0.0;
        ret_id_v = _lv;
        return ret_id_v;
    }


    inline double id_nai_localize_arguments(int id, int pnodecount, localize_arguments_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double _lnai) {
        double ret_id_nai = 0.0;
        ret_id_nai = _lnai;
        return ret_id_nai;
    }


    inline double id_ina_localize_arguments(int id, int pnodecount, localize_arguments_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double _lina) {
        double ret_id_ina = 0.0;
        ret_id_ina = _lina;
        return ret_id_ina;
    }


    inline double id_x_localize_arguments(int id, int pnodecount, localize_arguments_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double _lx) {
        double ret_id_x = 0.0;
        ret_id_x = _lx;
        return ret_id_x;
    }


    inline double id_g_localize_arguments(int id, int pnodecount, localize_arguments_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double _lg) {
        double ret_id_g = 0.0;
        ret_id_g = _lg;
        return ret_id_g;
    }


    inline double id_s_localize_arguments(int id, int pnodecount, localize_arguments_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double _ls) {
        double ret_id_s = 0.0;
        ret_id_s = _ls;
        return ret_id_s;
    }


    inline double id_p_localize_arguments(int id, int pnodecount, localize_arguments_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double _lp) {
        double ret_id_p = 0.0;
        ret_id_p = _lp;
        return ret_id_p;
    }


    /** initialize channel */
    void nrn_init_localize_arguments(NrnThread* nt, Memb_list* ml, int type) {
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;

        setup_instance(nt, ml);
        auto* const inst = static_cast<localize_arguments_Instance*>(ml->instance);

        if (_nrn_skip_initmodel == 0) {
            #pragma omp simd
            #pragma ivdep
            for (int id = 0; id < nodecount; id++) {
                int node_id = node_index[id];
                double v = voltage[node_id];
                #if NRN_PRCELLSTATE
                inst->v_unused[id] = v;
                #endif
                inst->ina[id] = inst->ion_ina[indexes[0*pnodecount + id]];
                inst->nai[id] = inst->ion_nai[indexes[1*pnodecount + id]];
                inst->s[id] = inst->global->s0;
                inst->x[id] = 42.0;
                inst->s[id] = 42.0;
            }
        }
    }


    /** register channel with the simulator */
    void _localize_arguments_reg() {

        int mech_type = nrn_get_mechtype("localize_arguments");
        localize_arguments_global.mech_type = mech_type;
        if (mech_type == -1) {
            return;
        }

        _nrn_layout_reg(mech_type, 0);
        register_mech(mechanism_info, nrn_alloc_localize_arguments, nullptr, nullptr, nullptr, nrn_init_localize_arguments, nrn_private_constructor_localize_arguments, nrn_private_destructor_localize_arguments, first_pointer_var_index(), 1);
        localize_arguments_global.na_type = nrn_get_mechtype("na_ion");

        hoc_register_prop_size(mech_type, float_variables_size(), int_variables_size());
        hoc_register_dparam_semantics(mech_type, 0, "na_ion");
        hoc_register_dparam_semantics(mech_type, 1, "na_ion");
        hoc_register_dparam_semantics(mech_type, 2, "na_ion");
        hoc_register_var(hoc_scalar_double, hoc_vector_double, NULL);
    }
}
