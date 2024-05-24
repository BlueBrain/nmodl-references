/*********************************************************
Model Name      : tbl
Filename        : table.mod
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
        "tbl",
        "e_tbl",
        "gmax_tbl",
        0,
        "g_tbl",
        "i_tbl",
        0,
        0,
        0
    };


    /** all global variables */
    struct tbl_Store {
        int reset{};
        int mech_type{};
        double k{0.1};
        double d{-50};
        double usetable{1};
        double tmin_sigmoid1{};
        double mfac_sigmoid1{};
        double t_sig[156]{};
    };
    static_assert(std::is_trivially_copy_constructible_v<tbl_Store>);
    static_assert(std::is_trivially_move_constructible_v<tbl_Store>);
    static_assert(std::is_trivially_copy_assignable_v<tbl_Store>);
    static_assert(std::is_trivially_move_assignable_v<tbl_Store>);
    static_assert(std::is_trivially_destructible_v<tbl_Store>);
    tbl_Store tbl_global;


    /** all mechanism instance variables and global variables */
    struct tbl_Instance  {
        const double* e{};
        const double* gmax{};
        double* g{};
        double* i{};
        double* sig{};
        double* v_unused{};
        double* g_unused{};
        tbl_Store* global{&tbl_global};
    };


    /** connect global (scalar) variables to hoc -- */
    static DoubScal hoc_scalar_double[] = {
        {"k_tbl", &tbl_global.k},
        {"d_tbl", &tbl_global.d},
        {"usetable_tbl", &tbl_global.usetable},
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
        return 7;
    }


    static inline int int_variables_size() {
        return 0;
    }


    static inline int get_mech_type() {
        return tbl_global.mech_type;
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
    static void nrn_private_constructor_tbl(NrnThread* nt, Memb_list* ml, int type) {
        assert(!ml->instance);
        assert(!ml->global_variables);
        assert(ml->global_variables_size == 0);
        auto* const inst = new tbl_Instance{};
        assert(inst->global == &tbl_global);
        ml->instance = inst;
        ml->global_variables = inst->global;
        ml->global_variables_size = sizeof(tbl_Store);
    }

    // Deallocate the instance structure
    static void nrn_private_destructor_tbl(NrnThread* nt, Memb_list* ml, int type) {
        auto* const inst = static_cast<tbl_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &tbl_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(tbl_Store));
        delete inst;
        ml->instance = nullptr;
        ml->global_variables = nullptr;
        ml->global_variables_size = 0;
    }

    /** initialize mechanism instance variables */
    static inline void setup_instance(NrnThread* nt, Memb_list* ml) {
        auto* const inst = static_cast<tbl_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &tbl_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(tbl_Store));
        int pnodecount = ml->_nodecount_padded;
        Datum* indexes = ml->pdata;
        inst->e = ml->data+0*pnodecount;
        inst->gmax = ml->data+1*pnodecount;
        inst->g = ml->data+2*pnodecount;
        inst->i = ml->data+3*pnodecount;
        inst->sig = ml->data+4*pnodecount;
        inst->v_unused = ml->data+5*pnodecount;
        inst->g_unused = ml->data+6*pnodecount;
    }



    static void nrn_alloc_tbl(double* data, Datum* indexes, int type) {
        // do nothing
    }


    void nrn_constructor_tbl(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<tbl_Instance*>(ml->instance);

        #endif
    }


    void nrn_destructor_tbl(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<tbl_Instance*>(ml->instance);

        #endif
    }


    inline int sigmoid1_tbl(int id, int pnodecount, tbl_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double arg_v);


    inline int f_sigmoid1_tbl(int id, int pnodecount, tbl_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double arg_v) {
        int ret_f_sigmoid1 = 0;
        inst->sig[id] = 1.0 / (1.0 + exp(inst->global->k * (arg_v - inst->global->d)));
        return ret_f_sigmoid1;
    }


    void lazy_update_sigmoid1_tbl(int id, int pnodecount, tbl_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v) {
        if (inst->global->usetable == 0) {
            return;
        }
        static bool make_table = true;
        static double save_k;
        static double save_d;
        if (save_k != inst->global->k) {
            make_table = true;
        }
        if (save_d != inst->global->d) {
            make_table = true;
        }
        if (make_table) {
            make_table = false;
            inst->global->tmin_sigmoid1 =  -127.0;
            double tmax = 128.0;
            double dx = (tmax-inst->global->tmin_sigmoid1) / 155.;
            inst->global->mfac_sigmoid1 = 1./dx;
            double x = inst->global->tmin_sigmoid1;
            for (std::size_t i = 0; i < 156; x += dx, i++) {
                f_sigmoid1_tbl(id, pnodecount, inst, data, indexes, thread, nt, v, x);
                inst->global->t_sig[i] = inst->sig[id];
            }
            save_k = inst->global->k;
            save_d = inst->global->d;
        }
    }


    inline int sigmoid1_tbl(int id, int pnodecount, tbl_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double arg_v){
        if (inst->global->usetable == 0) {
            f_sigmoid1_tbl(id, pnodecount, inst, data, indexes, thread, nt, v, arg_v);
            return 0;
        }
        double xi = inst->global->mfac_sigmoid1 * (arg_v - inst->global->tmin_sigmoid1);
        if (isnan(xi)) {
            inst->sig[id] = xi;
            return 0;
        }
        if (xi <= 0. || xi >= 155.) {
            int index = (xi <= 0.) ? 0 : 155;
            inst->sig[id] = inst->global->t_sig[index];
            return 0;
        }
        int i = int(xi);
        double theta = xi - double(i);
        inst->sig[id] = inst->global->t_sig[i] + theta*(inst->global->t_sig[i+1]-inst->global->t_sig[i]);
        return 0;
    }


    /** initialize channel */
    void nrn_init_tbl(NrnThread* nt, Memb_list* ml, int type) {
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;

        setup_instance(nt, ml);
        auto* const inst = static_cast<tbl_Instance*>(ml->instance);

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


    inline double nrn_current_tbl(int id, int pnodecount, tbl_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v) {
        double current = 0.0;
        sigmoid1_tbl(id, pnodecount, inst, data, indexes, thread, nt, v, v);
        inst->g[id] = inst->gmax[id] * inst->sig[id];
        inst->i[id] = inst->g[id] * (v - inst->e[id]);
        current += inst->i[id];
        return current;
    }


    /** update current */
    void nrn_cur_tbl(NrnThread* nt, Memb_list* ml, int type) {
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        double* vec_rhs = nt->_actual_rhs;
        double* vec_d = nt->_actual_d;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<tbl_Instance*>(ml->instance);

        #pragma omp simd
        #pragma ivdep
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_index[id];
            double v = voltage[node_id];
            #if NRN_PRCELLSTATE
            inst->v_unused[id] = v;
            #endif
            double g = nrn_current_tbl(id, pnodecount, inst, data, indexes, thread, nt, v+0.001);
            double rhs = nrn_current_tbl(id, pnodecount, inst, data, indexes, thread, nt, v);
            g = (g-rhs)/0.001;
            #if NRN_PRCELLSTATE
            inst->g_unused[id] = g;
            #endif
            vec_rhs[node_id] -= rhs;
            vec_d[node_id] += g;
        }
    }


    /** update state */
    void nrn_state_tbl(NrnThread* nt, Memb_list* ml, int type) {
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<tbl_Instance*>(ml->instance);

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


    static void check_table_thread_tbl (int id, int pnodecount, double* data, Datum* indexes, ThreadDatum* thread, NrnThread* nt, Memb_list* ml, int tml_id) {
        setup_instance(nt, ml);
        auto* const inst = static_cast<tbl_Instance*>(ml->instance);
        double v = 0;
        lazy_update_sigmoid1_tbl(id, pnodecount, inst, data, indexes, thread, nt, v);
    }


    /** register channel with the simulator */
    void _table_reg() {

        int mech_type = nrn_get_mechtype("tbl");
        tbl_global.mech_type = mech_type;
        if (mech_type == -1) {
            return;
        }

        _nrn_layout_reg(mech_type, 0);
        register_mech(mechanism_info, nrn_alloc_tbl, nrn_cur_tbl, nullptr, nrn_state_tbl, nrn_init_tbl, nrn_private_constructor_tbl, nrn_private_destructor_tbl, first_pointer_var_index(), 1);

        _nrn_thread_table_reg(mech_type, check_table_thread_tbl);
        hoc_register_prop_size(mech_type, float_variables_size(), int_variables_size());
        hoc_register_var(hoc_scalar_double, hoc_vector_double, NULL);
    }
}
