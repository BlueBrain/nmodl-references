/*********************************************************
Model Name      : hodhux
Filename        : hodhux.mod
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
        "hodhux",
        "gnabar_hodhux",
        "gkbar_hodhux",
        "gl_hodhux",
        "el_hodhux",
        0,
        "il_hodhux",
        "minf_hodhux",
        "hinf_hodhux",
        "ninf_hodhux",
        "mexp_hodhux",
        "hexp_hodhux",
        "nexp_hodhux",
        0,
        "m_hodhux",
        "h_hodhux",
        "n_hodhux",
        0,
        0
    };


    /** all global variables */
    struct hodhux_Store {
        int na_type{};
        int k_type{};
        double m0{};
        double h0{};
        double n0{};
        int reset{};
        int mech_type{};
    };
    static_assert(std::is_trivially_copy_constructible_v<hodhux_Store>);
    static_assert(std::is_trivially_move_constructible_v<hodhux_Store>);
    static_assert(std::is_trivially_copy_assignable_v<hodhux_Store>);
    static_assert(std::is_trivially_move_assignable_v<hodhux_Store>);
    static_assert(std::is_trivially_destructible_v<hodhux_Store>);
    hodhux_Store hodhux_global;


    /** all mechanism instance variables and global variables */
    struct hodhux_Instance  {
        double* celsius{&coreneuron::celsius};
        const double* gnabar{};
        const double* gkbar{};
        const double* gl{};
        const double* el{};
        double* il{};
        double* minf{};
        double* hinf{};
        double* ninf{};
        double* mexp{};
        double* hexp{};
        double* nexp{};
        double* m{};
        double* h{};
        double* n{};
        double* ena{};
        double* ek{};
        double* Dm{};
        double* Dh{};
        double* Dn{};
        double* ina{};
        double* ik{};
        double* v_unused{};
        double* g_unused{};
        const double* ion_ena{};
        double* ion_ina{};
        double* ion_dinadv{};
        const double* ion_ek{};
        double* ion_ik{};
        double* ion_dikdv{};
        hodhux_Store* global{&hodhux_global};
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
        return 23;
    }


    static inline int int_variables_size() {
        return 6;
    }


    static inline int get_mech_type() {
        return hodhux_global.mech_type;
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
    static void nrn_private_constructor_hodhux(NrnThread* nt, Memb_list* ml, int type) {
        assert(!ml->instance);
        assert(!ml->global_variables);
        assert(ml->global_variables_size == 0);
        auto* const inst = new hodhux_Instance{};
        assert(inst->global == &hodhux_global);
        ml->instance = inst;
        ml->global_variables = inst->global;
        ml->global_variables_size = sizeof(hodhux_Store);
    }

    // Deallocate the instance structure
    static void nrn_private_destructor_hodhux(NrnThread* nt, Memb_list* ml, int type) {
        auto* const inst = static_cast<hodhux_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &hodhux_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(hodhux_Store));
        delete inst;
        ml->instance = nullptr;
        ml->global_variables = nullptr;
        ml->global_variables_size = 0;
    }

    /** initialize mechanism instance variables */
    static inline void setup_instance(NrnThread* nt, Memb_list* ml) {
        auto* const inst = static_cast<hodhux_Instance*>(ml->instance);
        assert(inst);
        assert(inst->global);
        assert(inst->global == &hodhux_global);
        assert(inst->global == ml->global_variables);
        assert(ml->global_variables_size == sizeof(hodhux_Store));
        int pnodecount = ml->_nodecount_padded;
        Datum* indexes = ml->pdata;
        inst->gnabar = ml->data+0*pnodecount;
        inst->gkbar = ml->data+1*pnodecount;
        inst->gl = ml->data+2*pnodecount;
        inst->el = ml->data+3*pnodecount;
        inst->il = ml->data+4*pnodecount;
        inst->minf = ml->data+5*pnodecount;
        inst->hinf = ml->data+6*pnodecount;
        inst->ninf = ml->data+7*pnodecount;
        inst->mexp = ml->data+8*pnodecount;
        inst->hexp = ml->data+9*pnodecount;
        inst->nexp = ml->data+10*pnodecount;
        inst->m = ml->data+11*pnodecount;
        inst->h = ml->data+12*pnodecount;
        inst->n = ml->data+13*pnodecount;
        inst->ena = ml->data+14*pnodecount;
        inst->ek = ml->data+15*pnodecount;
        inst->Dm = ml->data+16*pnodecount;
        inst->Dh = ml->data+17*pnodecount;
        inst->Dn = ml->data+18*pnodecount;
        inst->ina = ml->data+19*pnodecount;
        inst->ik = ml->data+20*pnodecount;
        inst->v_unused = ml->data+21*pnodecount;
        inst->g_unused = ml->data+22*pnodecount;
        inst->ion_ena = nt->_data;
        inst->ion_ina = nt->_data;
        inst->ion_dinadv = nt->_data;
        inst->ion_ek = nt->_data;
        inst->ion_ik = nt->_data;
        inst->ion_dikdv = nt->_data;
    }



    static void nrn_alloc_hodhux(double* data, Datum* indexes, int type) {
        // do nothing
    }


    void nrn_constructor_hodhux(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<hodhux_Instance*>(ml->instance);

        #endif
    }


    void nrn_destructor_hodhux(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<hodhux_Instance*>(ml->instance);

        #endif
    }


    inline double vtrap_hodhux(int id, int pnodecount, hodhux_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double x, double y);
    inline int states_hodhux(int id, int pnodecount, hodhux_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v);
    inline int rates_hodhux(int id, int pnodecount, hodhux_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double arg_v);


    inline int states_hodhux(int id, int pnodecount, hodhux_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v) {
        int ret_states = 0;
        rates_hodhux(id, pnodecount, inst, data, indexes, thread, nt, v, v);
        inst->m[id] = inst->m[id] + inst->mexp[id] * (inst->minf[id] - inst->m[id]);
        inst->h[id] = inst->h[id] + inst->hexp[id] * (inst->hinf[id] - inst->h[id]);
        inst->n[id] = inst->n[id] + inst->nexp[id] * (inst->ninf[id] - inst->n[id]);
        return ret_states;
    }


    inline int rates_hodhux(int id, int pnodecount, hodhux_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double arg_v) {
        int ret_rates = 0;
        double q10, tinc, alpha, beta, sum;
        q10 = pow(3.0, ((*(inst->celsius) - 6.3) / 10.0));
        tinc =  -nt->_dt * q10;
        alpha = .1 * vtrap_hodhux(id, pnodecount, inst, data, indexes, thread, nt, v,  -(arg_v + 40.0), 10.0);
        beta = 4.0 * exp( -(arg_v + 65.0) / 18.0);
        sum = alpha + beta;
        inst->minf[id] = alpha / sum;
        inst->mexp[id] = 1.0 - exp(tinc * sum);
        alpha = .07 * exp( -(arg_v + 65.0) / 20.0);
        beta = 1.0 / (exp( -(arg_v + 35.0) / 10.0) + 1.0);
        sum = alpha + beta;
        inst->hinf[id] = alpha / sum;
        inst->hexp[id] = 1.0 - exp(tinc * sum);
        alpha = .01 * vtrap_hodhux(id, pnodecount, inst, data, indexes, thread, nt, v,  -(arg_v + 55.0), 10.0);
        beta = .125 * exp( -(arg_v + 65.0) / 80.0);
        sum = alpha + beta;
        inst->ninf[id] = alpha / sum;
        inst->nexp[id] = 1.0 - exp(tinc * sum);
        return ret_rates;
    }


    inline double vtrap_hodhux(int id, int pnodecount, hodhux_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double x, double y) {
        double ret_vtrap = 0.0;
        if (fabs(x / y) < 1e-6) {
            ret_vtrap = y * (1.0 - x / y / 2.0);
        } else {
            ret_vtrap = x / (exp(x / y) - 1.0);
        }
        return ret_vtrap;
    }


    /** initialize channel */
    void nrn_init_hodhux(NrnThread* nt, Memb_list* ml, int type) {
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;

        setup_instance(nt, ml);
        auto* const inst = static_cast<hodhux_Instance*>(ml->instance);

        if (_nrn_skip_initmodel == 0) {
            #pragma omp simd
            #pragma ivdep
            for (int id = 0; id < nodecount; id++) {
                int node_id = node_index[id];
                double v = voltage[node_id];
                #if NRN_PRCELLSTATE
                inst->v_unused[id] = v;
                #endif
                inst->ena[id] = inst->ion_ena[indexes[0*pnodecount + id]];
                inst->ek[id] = inst->ion_ek[indexes[3*pnodecount + id]];
                inst->m[id] = inst->global->m0;
                inst->h[id] = inst->global->h0;
                inst->n[id] = inst->global->n0;
                rates_hodhux(id, pnodecount, inst, data, indexes, thread, nt, v, v);
                inst->m[id] = inst->minf[id];
                inst->h[id] = inst->hinf[id];
                inst->n[id] = inst->ninf[id];
            }
        }
    }


    inline double nrn_current_hodhux(int id, int pnodecount, hodhux_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v) {
        double current = 0.0;
        inst->ina[id] = inst->gnabar[id] * inst->m[id] * inst->m[id] * inst->m[id] * inst->h[id] * (v - inst->ena[id]);
        inst->ik[id] = inst->gkbar[id] * inst->n[id] * inst->n[id] * inst->n[id] * inst->n[id] * (v - inst->ek[id]);
        inst->il[id] = inst->gl[id] * (v - inst->el[id]);
        current += inst->il[id];
        current += inst->ina[id];
        current += inst->ik[id];
        return current;
    }


    /** update current */
    void nrn_cur_hodhux(NrnThread* nt, Memb_list* ml, int type) {
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        double* vec_rhs = nt->_actual_rhs;
        double* vec_d = nt->_actual_d;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<hodhux_Instance*>(ml->instance);

        #pragma omp simd
        #pragma ivdep
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_index[id];
            double v = voltage[node_id];
            #if NRN_PRCELLSTATE
            inst->v_unused[id] = v;
            #endif
            inst->ena[id] = inst->ion_ena[indexes[0*pnodecount + id]];
            inst->ek[id] = inst->ion_ek[indexes[3*pnodecount + id]];
            double g = nrn_current_hodhux(id, pnodecount, inst, data, indexes, thread, nt, v+0.001);
            double dina = inst->ina[id];
            double dik = inst->ik[id];
            double rhs = nrn_current_hodhux(id, pnodecount, inst, data, indexes, thread, nt, v);
            g = (g-rhs)/0.001;
            inst->ion_dinadv[indexes[2*pnodecount + id]] += (dina-inst->ina[id])/0.001;
            inst->ion_dikdv[indexes[5*pnodecount + id]] += (dik-inst->ik[id])/0.001;
            inst->ion_ina[indexes[1*pnodecount + id]] += inst->ina[id];
            inst->ion_ik[indexes[4*pnodecount + id]] += inst->ik[id];
            #if NRN_PRCELLSTATE
            inst->g_unused[id] = g;
            #endif
            vec_rhs[node_id] -= rhs;
            vec_d[node_id] += g;
        }
    }


    /** update state */
    void nrn_state_hodhux(NrnThread* nt, Memb_list* ml, int type) {
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        auto* const inst = static_cast<hodhux_Instance*>(ml->instance);

        #pragma omp simd
        #pragma ivdep
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_index[id];
            double v = voltage[node_id];
            #if NRN_PRCELLSTATE
            inst->v_unused[id] = v;
            #endif
            inst->ena[id] = inst->ion_ena[indexes[0*pnodecount + id]];
            inst->ek[id] = inst->ion_ek[indexes[3*pnodecount + id]];
            rates_hodhux(id, pnodecount, inst, data, indexes, thread, nt, v, v);
            inst->m[id] = inst->m[id] + inst->mexp[id] * (inst->minf[id] - inst->m[id]);
            inst->h[id] = inst->h[id] + inst->hexp[id] * (inst->hinf[id] - inst->h[id]);
            inst->n[id] = inst->n[id] + inst->nexp[id] * (inst->ninf[id] - inst->n[id]);
        }
    }


    /** register channel with the simulator */
    void _hodhux_reg() {

        int mech_type = nrn_get_mechtype("hodhux");
        hodhux_global.mech_type = mech_type;
        if (mech_type == -1) {
            return;
        }

        register_mech(mechanism_info, nrn_alloc_hodhux, nrn_cur_hodhux, nullptr, nrn_state_hodhux, nrn_init_hodhux, nrn_private_constructor_hodhux, nrn_private_destructor_hodhux, first_pointer_var_index(), 1);
        hodhux_global.na_type = nrn_get_mechtype("na_ion");
        hodhux_global.k_type = nrn_get_mechtype("k_ion");

        hoc_register_prop_size(mech_type, float_variables_size(), int_variables_size());
        hoc_register_dparam_semantics(mech_type, 0, "na_ion");
        hoc_register_dparam_semantics(mech_type, 1, "na_ion");
        hoc_register_dparam_semantics(mech_type, 2, "na_ion");
        hoc_register_dparam_semantics(mech_type, 3, "k_ion");
        hoc_register_dparam_semantics(mech_type, 4, "k_ion");
        hoc_register_dparam_semantics(mech_type, 5, "k_ion");
        hoc_register_var(hoc_scalar_double, hoc_vector_double, NULL);
    }
}
