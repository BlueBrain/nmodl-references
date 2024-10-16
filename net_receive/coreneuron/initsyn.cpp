/*********************************************************
Model Name      : InitSyn
Filename        : initsyn.mod
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
        "InitSyn",
        0,
        "a0",
        0,
        0,
        0
    };


    /** all global variables */
    struct InitSyn_Store {
        int point_type{};
        int reset{};
        int mech_type{};
    };
    static_assert(std::is_trivially_copy_constructible_v<InitSyn_Store>);
    static_assert(std::is_trivially_move_constructible_v<InitSyn_Store>);
    static_assert(std::is_trivially_copy_assignable_v<InitSyn_Store>);
    static_assert(std::is_trivially_move_assignable_v<InitSyn_Store>);
    static_assert(std::is_trivially_destructible_v<InitSyn_Store>);
    static InitSyn_Store InitSyn_global;


    /** all mechanism instance variables and global variables */
    struct InitSyn_Instance  {
        double* a0{};
        double* v_unused{};
        double* tsave{};
        const double* node_area{};
        const int* point_process{};
        InitSyn_Store* global{&InitSyn_global};
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
        return 2;
    }


    static inline int float_variables_size() {
        return 3;
    }


    static inline int int_variables_size() {
        return 2;
    }


    static inline int get_mech_type() {
        return InitSyn_global.mech_type;
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
    static void nrn_private_constructor_InitSyn(NrnThread* nt, Memb_list* ml, int type) {
        assert(!ml->instance);
        assert(!ml->global_variables);
        assert(ml->global_variables_size == 0);
        auto* const inst = new InitSyn_Instance{};
        assert(inst->global == &InitSyn_global);
        ml->instance = inst;
        ml->global_variables = inst->global;
        ml->global_variables_size = sizeof(InitSyn_Store);
    }

    // Deallocate the instance structure
    static void nrn_private_destructor_InitSyn(NrnThread* nt, Memb_list* ml, int type) {
        auto* const inst = static_cast<InitSyn_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &InitSyn_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(InitSyn_Store));
        delete inst;
        ml->instance = nullptr;
        ml->global_variables = nullptr;
        ml->global_variables_size = 0;
    }

    /** initialize mechanism instance variables */
    static inline void setup_instance(NrnThread* nt, Memb_list* ml) {
        auto* const inst = static_cast<InitSyn_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &InitSyn_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(InitSyn_Store));
        int pnodecount = ml->_nodecount_padded;
        Datum* indexes = ml->pdata;
        inst->a0 = ml->data+0*pnodecount;
        inst->v_unused = ml->data+1*pnodecount;
        inst->tsave = ml->data+2*pnodecount;
        inst->node_area = nt->_data;
        inst->point_process = ml->pdata;
    }



    static void nrn_alloc_InitSyn(double* data, Datum* indexes, int type) {
        // do nothing
    }


    void nrn_constructor_InitSyn(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<InitSyn_Instance*>(ml->instance);

        #endif
    }


    void nrn_destructor_InitSyn(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<InitSyn_Instance*>(ml->instance);

        #endif
    }


    /** initialize block for net receive */
    static void net_init(Point_process* pnt, int weight_index, double flag) {
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
        auto* const inst = static_cast<InitSyn_Instance*>(ml->instance);

        double* a = weights + weight_index + 1;
        (*a) = inst->a0[id];
        auto& nsb = ml->_net_send_buffer;
    }


    static inline void net_receive_kernel_InitSyn(double t, Point_process* pnt, InitSyn_Instance* inst, NrnThread* nt, Memb_list* ml, int weight_index, double flag) {
        int tid = pnt->_tid;
        int id = pnt->_i_instance;
        double v = 0;
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        double* data = ml->data;
        double* weights = nt->weights;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;

        double* a = weights + weight_index + 1;
        inst->tsave[id] = t;
        {
        }
    }


    static void net_receive_InitSyn(Point_process* pnt, int weight_index, double flag) {
        NrnThread* nt = nrn_threads + pnt->_tid;
        Memb_list* ml = get_memb_list(nt);
        NetReceiveBuffer_t* nrb = ml->_net_receive_buffer;
        if (nrb->_cnt >= nrb->_size) {
            realloc_net_receive_buffer(nt, ml);
        }
        int id = nrb->_cnt;
        nrb->_pnt_index[id] = pnt-nt->pntprocs;
        nrb->_weight_index[id] = weight_index;
        nrb->_nrb_t[id] = nt->_t;
        nrb->_nrb_flag[id] = flag;
        nrb->_cnt++;
    }


    void net_buf_receive_InitSyn(NrnThread* nt) {
        Memb_list* ml = get_memb_list(nt);
        if (!ml) {
            return;
        }

        NetReceiveBuffer_t* nrb = ml->_net_receive_buffer;
        auto* const inst = static_cast<InitSyn_Instance*>(ml->instance);
        int count = nrb->_displ_cnt;
        #pragma omp simd
        #pragma ivdep
        for (int i = 0; i < count; i++) {
            int start = nrb->_displ[i];
            int end = nrb->_displ[i+1];
            for (int j = start; j < end; j++) {
                int index = nrb->_nrb_index[j];
                int offset = nrb->_pnt_index[index];
                double t = nrb->_nrb_t[index];
                int weight_index = nrb->_weight_index[index];
                double flag = nrb->_nrb_flag[index];
                Point_process* point_process = nt->pntprocs + offset;
                net_receive_kernel_InitSyn(t, point_process, inst, nt, ml, weight_index, flag);
            }
        }
        nrb->_displ_cnt = 0;
        nrb->_cnt = 0;
    }


    /** initialize channel */
    void nrn_init_InitSyn(NrnThread* nt, Memb_list* ml, int type) {
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;

        setup_instance(nt, ml);
        auto* const inst = static_cast<InitSyn_Instance*>(ml->instance);

        if (_nrn_skip_initmodel == 0) {
            #pragma omp simd
            #pragma ivdep
            for (int id = 0; id < nodecount; id++) {
                inst->tsave[id] = -1e20;
                int node_id = node_index[id];
                double v = voltage[node_id];
                #if NRN_PRCELLSTATE
                inst->v_unused[id] = v;
                #endif
            }
        }
    }


    /** register channel with the simulator */
    void _initsyn_reg() {

        int mech_type = nrn_get_mechtype("InitSyn");
        InitSyn_global.mech_type = mech_type;
        if (mech_type == -1) {
            return;
        }

        _nrn_layout_reg(mech_type, 0);
        point_register_mech(mechanism_info, nrn_alloc_InitSyn, nullptr, nullptr, nullptr, nrn_init_InitSyn, nrn_private_constructor_InitSyn, nrn_private_destructor_InitSyn, first_pointer_var_index(), nullptr, nullptr, 1);

        hoc_register_prop_size(mech_type, float_variables_size(), int_variables_size());
        hoc_register_dparam_semantics(mech_type, 0, "area");
        hoc_register_dparam_semantics(mech_type, 1, "pntproc");
        hoc_register_net_receive_buffering(net_buf_receive_InitSyn, mech_type);
        set_pnt_receive(mech_type, net_receive_InitSyn, net_init, num_net_receive_args());
        hoc_register_var(hoc_scalar_double, hoc_vector_double, NULL);
    }
}
