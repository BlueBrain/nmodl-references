/*********************************************************
Model Name      : art_spiker
Filename        : art_spiker.mod
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
        "art_spiker",
        0,
        "y",
        "z",
        0,
        0,
        0
    };


    /** all global variables */
    struct art_spiker_Store {
        int point_type{};
        int reset{};
        int mech_type{};
    };
    static_assert(std::is_trivially_copy_constructible_v<art_spiker_Store>);
    static_assert(std::is_trivially_move_constructible_v<art_spiker_Store>);
    static_assert(std::is_trivially_copy_assignable_v<art_spiker_Store>);
    static_assert(std::is_trivially_move_assignable_v<art_spiker_Store>);
    static_assert(std::is_trivially_destructible_v<art_spiker_Store>);
    art_spiker_Store art_spiker_global;


    /** all mechanism instance variables and global variables */
    struct art_spiker_Instance  {
        double* y{};
        double* z{};
        double* v_unused{};
        double* tsave{};
        const double* node_area{};
        void** point_process{};
        void** tqitem{};
        art_spiker_Store* global{&art_spiker_global};
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


    static inline int num_net_receive_args() {
        return 1;
    }


    static inline int float_variables_size() {
        return 4;
    }


    static inline int int_variables_size() {
        return 3;
    }


    static inline int get_mech_type() {
        return art_spiker_global.mech_type;
    }


    static inline Memb_list* get_memb_list(NrnThread* nt) {
        if (!nt->_ml_list) {
            return nullptr;
        }
        return nt->_ml_list[get_mech_type()];
    }


    static inline void* mem_alloc(size_t num, size_t size, size_t alignment = 32) {
        size_t aligned_size = (num*size + alignment - 1) / alignment) * alignment;
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
    static void nrn_private_constructor_art_spiker(NrnThread* nt, Memb_list* ml, int type) {
        assert(!ml->instance);
        assert(!ml->global_variables);
        assert(ml->global_variables_size == 0);
        auto* const inst = new art_spiker_Instance{};
        assert(inst->global == &art_spiker_global);
        ml->instance = inst;
        ml->global_variables = inst->global;
        ml->global_variables_size = sizeof(art_spiker_Store);
    }

    // Deallocate the instance structure
    static void nrn_private_destructor_art_spiker(NrnThread* nt, Memb_list* ml, int type) {
        auto* const inst = static_cast<art_spiker_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &art_spiker_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(art_spiker_Store));
        delete inst;
        ml->instance = nullptr;
        ml->global_variables = nullptr;
        ml->global_variables_size = 0;
    }

    /** initialize mechanism instance variables */
    static inline void setup_instance(NrnThread* nt, Memb_list* ml) {
        auto* const inst = static_cast<art_spiker_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &art_spiker_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(art_spiker_Store));
        int pnodecount = ml->_nodecount_padded;
        Datum* indexes = ml->pdata;
        inst->y = ml->data+0*pnodecount;
        inst->z = ml->data+1*pnodecount;
        inst->v_unused = ml->data+2*pnodecount;
        inst->tsave = ml->data+3*pnodecount;
        inst->node_area = nt->_data;
        inst->point_process = nt->_vdata;
        inst->tqitem = nt->_vdata;
    }



    static void nrn_alloc_art_spiker(double* data, Datum* indexes, int type) {
        // do nothing
    }


    void nrn_constructor_art_spiker(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<art_spiker_Instance*>(ml->instance);

        #endif
    }


    void nrn_destructor_art_spiker(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<art_spiker_Instance*>(ml->instance);

        #endif
    }


    static inline void net_receive_art_spiker(Point_process* pnt, int weight_index, double flag) {
        int tid = pnt->_tid;
        int id = pnt->_i_instance;
        double v = 0;
        NrnThread* nt = nrn_threads + tid;
        Memb_list* ml = nt->_ml_list[pnt->_type];
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        double* data = ml->data;
        double* weights = nt->weights;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<art_spiker_Instance*>(ml->instance);

        double t = nt->_t;
        inst->tsave[id] = t;
        {
            if (flag == 0.0) {
                inst->y[id] = inst->y[id] + 1.0;
                artcell_net_move(&inst->tqitem[indexes[2*pnodecount + id]], pnt, t + 0.1);
            } else {
                inst->z[id] = inst->z[id] + 1.0;
                artcell_net_send(&inst->tqitem[indexes[2*pnodecount + id]], weight_index, pnt, nt->_t+2.0, 1.0);
            }
        }
    }


    /** initialize channel */
    void nrn_init_art_spiker(NrnThread* nt, Memb_list* ml, int type) {
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;

        setup_instance(nt, ml);
        auto* const inst = static_cast<art_spiker_Instance*>(ml->instance);

        if (_nrn_skip_initmodel == 0) {
            #pragma omp simd
            #pragma ivdep
            for (int id = 0; id < nodecount; id++) {
                inst->tsave[id] = -1e20;
                double v = 0.0;
                inst->y[id] = 0.0;
                inst->z[id] = 0.0;
                artcell_net_send(&inst->tqitem[indexes[2*pnodecount + id]], 0, (Point_process*)inst->point_process[indexes[1*pnodecount + id]], nt->_t+1.8, 1.0);
            }
        }
    }


    /** register channel with the simulator */
    void _art_spiker_reg() {

        int mech_type = nrn_get_mechtype("art_spiker");
        art_spiker_global.mech_type = mech_type;
        if (mech_type == -1) {
            return;
        }

        _nrn_layout_reg(mech_type, 0);
        point_register_mech(mechanism_info, nrn_alloc_art_spiker, nullptr, nullptr, nullptr, nrn_init_art_spiker, nrn_private_constructor_art_spiker, nrn_private_destructor_art_spiker, first_pointer_var_index(), nullptr, nullptr, 1);

        hoc_register_prop_size(mech_type, float_variables_size(), int_variables_size());
        hoc_register_dparam_semantics(mech_type, 0, "area");
        hoc_register_dparam_semantics(mech_type, 1, "pntproc");
        hoc_register_dparam_semantics(mech_type, 2, "netsend");
        add_nrn_artcell(mech_type, 2);
        set_pnt_receive(mech_type, net_receive_art_spiker, nullptr, num_net_receive_args());
        hoc_register_net_send_buffering(mech_type);
        hoc_register_var(hoc_scalar_double, hoc_vector_double, NULL);
    }
}
