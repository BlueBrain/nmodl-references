/*********************************************************
Model Name      : tbl_point_process
Filename        : table_point_process.mod
NMODL Version   : 7.7.0
Vectorized      : true
Threadsafe      : true
Created         : DATE
Simulator       : NEURON
Backend         : C++ (api-compatibility)
NMODL Compiler  : VERSION
*********************************************************/

#include <Eigen/Dense>
#include <Eigen/LU>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "mech_api.h"
#include "neuron/cache/mechanism_range.hpp"
#include "nrniv_mf.h"
#include "section_fwd.hpp"

/* NEURON global macro definitions */
/* VECTORIZED */
#define NRN_VECTORIZED 1

static constexpr auto number_of_datum_variables = 2;
static constexpr auto number_of_floating_point_variables = 7;

namespace {
template <typename T>
using _nrn_mechanism_std_vector = std::vector<T>;
using _nrn_model_sorted_token = neuron::model_sorted_token;
using _nrn_mechanism_cache_range = neuron::cache::MechanismRange<number_of_floating_point_variables, number_of_datum_variables>;
using _nrn_mechanism_cache_instance = neuron::cache::MechanismInstance<number_of_floating_point_variables, number_of_datum_variables>;
using _nrn_non_owning_id_without_container = neuron::container::non_owning_identifier_without_container;
template <typename T>
using _nrn_mechanism_field = neuron::mechanism::field<T>;
template <typename... Args>
void _nrn_mechanism_register_data_fields(Args&&... args) {
    neuron::mechanism::register_data_fields(std::forward<Args>(args)...);
}
}  // namespace

extern Prop* nrn_point_prop_;
void _nrn_thread_table_reg(int, nrn_thread_table_check_t);
extern void _cvode_abstol(Symbol**, double*, int);
extern Node* nrn_alloc_node_;


namespace neuron {
    #ifndef NRN_PRCELLSTATE
    #define NRN_PRCELLSTATE 0
    #endif


    /** channel information */
    static const char *mechanism_info[] = {
        "7.7.0",
        "tbl_point_process",
        0,
        "g",
        "i",
        "v1",
        "v2",
        0,
        0,
        0
    };


    /* NEURON global variables */
    static int mech_type;
    static int _pointtype;
    static _nrn_mechanism_std_vector<Datum> _extcall_thread;


    /** all global variables */
    struct tbl_point_process_Store {
        double k{0.1};
        double d{-50};
        double c1{1};
        double c2{2};
        double usetable{1};
        double tmin_sigmoidal{};
        double mfac_sigmoidal{};
        double tmin_quadratic{};
        double mfac_quadratic{};
        double tmin_sinusoidal{};
        double mfac_sinusoidal{};
        double t_v1[801]{};
        double t_v2[801]{};
        double t_sig[156]{};
        double t_quadratic[501]{};
    };
    static_assert(std::is_trivially_copy_constructible_v<tbl_point_process_Store>);
    static_assert(std::is_trivially_move_constructible_v<tbl_point_process_Store>);
    static_assert(std::is_trivially_copy_assignable_v<tbl_point_process_Store>);
    static_assert(std::is_trivially_move_assignable_v<tbl_point_process_Store>);
    static_assert(std::is_trivially_destructible_v<tbl_point_process_Store>);
    static tbl_point_process_Store tbl_point_process_global;
    auto k_tbl_point_process() -> std::decay<decltype(tbl_point_process_global.k)>::type  {
        return tbl_point_process_global.k;
    }
    auto d_tbl_point_process() -> std::decay<decltype(tbl_point_process_global.d)>::type  {
        return tbl_point_process_global.d;
    }
    auto c1_tbl_point_process() -> std::decay<decltype(tbl_point_process_global.c1)>::type  {
        return tbl_point_process_global.c1;
    }
    auto c2_tbl_point_process() -> std::decay<decltype(tbl_point_process_global.c2)>::type  {
        return tbl_point_process_global.c2;
    }
    auto usetable_tbl_point_process() -> std::decay<decltype(tbl_point_process_global.usetable)>::type  {
        return tbl_point_process_global.usetable;
    }
    auto tmin_sigmoidal_tbl_point_process() -> std::decay<decltype(tbl_point_process_global.tmin_sigmoidal)>::type  {
        return tbl_point_process_global.tmin_sigmoidal;
    }
    auto mfac_sigmoidal_tbl_point_process() -> std::decay<decltype(tbl_point_process_global.mfac_sigmoidal)>::type  {
        return tbl_point_process_global.mfac_sigmoidal;
    }
    auto tmin_quadratic_tbl_point_process() -> std::decay<decltype(tbl_point_process_global.tmin_quadratic)>::type  {
        return tbl_point_process_global.tmin_quadratic;
    }
    auto mfac_quadratic_tbl_point_process() -> std::decay<decltype(tbl_point_process_global.mfac_quadratic)>::type  {
        return tbl_point_process_global.mfac_quadratic;
    }
    auto tmin_sinusoidal_tbl_point_process() -> std::decay<decltype(tbl_point_process_global.tmin_sinusoidal)>::type  {
        return tbl_point_process_global.tmin_sinusoidal;
    }
    auto mfac_sinusoidal_tbl_point_process() -> std::decay<decltype(tbl_point_process_global.mfac_sinusoidal)>::type  {
        return tbl_point_process_global.mfac_sinusoidal;
    }
    auto t_v1_tbl_point_process() -> std::decay<decltype(tbl_point_process_global.t_v1)>::type  {
        return tbl_point_process_global.t_v1;
    }
    auto t_v2_tbl_point_process() -> std::decay<decltype(tbl_point_process_global.t_v2)>::type  {
        return tbl_point_process_global.t_v2;
    }
    auto t_sig_tbl_point_process() -> std::decay<decltype(tbl_point_process_global.t_sig)>::type  {
        return tbl_point_process_global.t_sig;
    }
    auto t_quadratic_tbl_point_process() -> std::decay<decltype(tbl_point_process_global.t_quadratic)>::type  {
        return tbl_point_process_global.t_quadratic;
    }

    static std::vector<double> _parameter_defaults = {
    };


    /** all mechanism instance variables and global variables */
    struct tbl_point_process_Instance  {
        double* g{};
        double* i{};
        double* v1{};
        double* v2{};
        double* sig{};
        double* v_unused{};
        double* g_unused{};
        const double* const* node_area{};
        tbl_point_process_Store* global{&tbl_point_process_global};
    };


    struct tbl_point_process_NodeData  {
        int const * nodeindices;
        double const * node_voltages;
        double * node_diagonal;
        double * node_rhs;
        int nodecount;
    };


    static tbl_point_process_Instance make_instance_tbl_point_process(_nrn_mechanism_cache_range& _lmc) {
        return tbl_point_process_Instance {
            _lmc.template fpfield_ptr<0>(),
            _lmc.template fpfield_ptr<1>(),
            _lmc.template fpfield_ptr<2>(),
            _lmc.template fpfield_ptr<3>(),
            _lmc.template fpfield_ptr<4>(),
            _lmc.template fpfield_ptr<5>(),
            _lmc.template fpfield_ptr<6>(),
            _lmc.template dptr_field_ptr<0>()
        };
    }


    static tbl_point_process_NodeData make_node_data_tbl_point_process(NrnThread& nt, Memb_list& _ml_arg) {
        return tbl_point_process_NodeData {
            _ml_arg.nodeindices,
            nt.node_voltage_storage(),
            nt.node_d_storage(),
            nt.node_rhs_storage(),
            _ml_arg.nodecount
        };
    }
    static tbl_point_process_NodeData make_node_data_tbl_point_process(Prop * _prop) {
        static std::vector<int> node_index{0};
        Node* _node = _nrn_mechanism_access_node(_prop);
        return tbl_point_process_NodeData {
            node_index.data(),
            &_nrn_mechanism_access_voltage(_node),
            &_nrn_mechanism_access_d(_node),
            &_nrn_mechanism_access_rhs(_node),
            1
        };
    }

    static void nrn_destructor_tbl_point_process(Prop* prop);


    static void nrn_alloc_tbl_point_process(Prop* _prop) {
        Datum *_ppvar = nullptr;
        if (nrn_point_prop_) {
            _nrn_mechanism_access_alloc_seq(_prop) = _nrn_mechanism_access_alloc_seq(nrn_point_prop_);
            _ppvar = _nrn_mechanism_access_dparam(nrn_point_prop_);
        } else {
            _ppvar = nrn_prop_datum_alloc(mech_type, 2, _prop);
            _nrn_mechanism_access_dparam(_prop) = _ppvar;
            _nrn_mechanism_cache_instance _lmc{_prop};
            size_t const _iml = 0;
            assert(_nrn_mechanism_get_num_vars(_prop) == 7);
            /*initialize range parameters*/
        }
        _nrn_mechanism_access_dparam(_prop) = _ppvar;
        if(!nrn_point_prop_) {
        }
    }


    /* Mechanism procedures and functions */
    inline static double quadratic_tbl_point_process(_nrn_mechanism_cache_range& _lmc, tbl_point_process_Instance& inst, tbl_point_process_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _lx);
    inline static int sigmoidal_tbl_point_process(_nrn_mechanism_cache_range& _lmc, tbl_point_process_Instance& inst, tbl_point_process_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _lv);
    inline static int sinusoidal_tbl_point_process(_nrn_mechanism_cache_range& _lmc, tbl_point_process_Instance& inst, tbl_point_process_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _lx);
    static void _apply_diffusion_function(ldifusfunc2_t _f, const _nrn_model_sorted_token& _sorted_token, NrnThread& _nt) {
    }

    /* Point Process specific functions */
    static void* _hoc_create_pnt(Object* _ho) {
        return create_point_process(_pointtype, _ho);
    }
    static void _hoc_destroy_pnt(void* _vptr) {
        destroy_point_process(_vptr);
    }
    static double _hoc_loc_pnt(void* _vptr) {
        return loc_point_process(_pointtype, _vptr);
    }
    static double _hoc_has_loc(void* _vptr) {
        return has_loc_point(_vptr);
    }
    static double _hoc_get_loc_pnt(void* _vptr) {
        return (get_loc_point_process(_vptr));
    }
    /* Neuron setdata functions */
    extern void _nrn_setdata_reg(int, void(*)(Prop*));
    static void _setdata(Prop* _prop) {
    }
    static void _hoc_setdata(void* _vptr) {
        Prop* _prop;
        _prop = ((Point_process*)_vptr)->prop;
        _setdata(_prop);
    }
    void update_table_sigmoidal_tbl_point_process(_nrn_mechanism_cache_range& _lmc, tbl_point_process_Instance& inst, tbl_point_process_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt);
    void update_table_quadratic_tbl_point_process(_nrn_mechanism_cache_range& _lmc, tbl_point_process_Instance& inst, tbl_point_process_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt);
    void update_table_sinusoidal_tbl_point_process(_nrn_mechanism_cache_range& _lmc, tbl_point_process_Instance& inst, tbl_point_process_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt);
    static void _check_table_thread(Memb_list* _ml, size_t id, Datum* _ppvar, Datum* _thread, double* _globals, NrnThread* nt, int _type, const _nrn_model_sorted_token& _sorted_token)
{
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml, _type};
        auto inst = make_instance_tbl_point_process(_lmc);
        auto node_data = make_node_data_tbl_point_process(*nt, *_ml);
        update_table_sigmoidal_tbl_point_process(_lmc, inst, node_data, id, _ppvar, _thread, nt);
        update_table_quadratic_tbl_point_process(_lmc, inst, node_data, id, _ppvar, _thread, nt);
        update_table_sinusoidal_tbl_point_process(_lmc, inst, node_data, id, _ppvar, _thread, nt);
    }


    /** connect global (scalar) variables to hoc -- */
    static DoubScal hoc_scalar_double[] = {
        {"k_tbl_point_process", &tbl_point_process_global.k},
        {"d_tbl_point_process", &tbl_point_process_global.d},
        {"c1_tbl_point_process", &tbl_point_process_global.c1},
        {"c2_tbl_point_process", &tbl_point_process_global.c2},
        {"usetable_tbl_point_process", &tbl_point_process_global.usetable},
        {nullptr, nullptr}
    };


    /** connect global (array) variables to hoc -- */
    static DoubVec hoc_vector_double[] = {
        {nullptr, nullptr, 0}
    };


    /* declaration of user functions */
    static double _hoc_quadratic(void * _vptr);
    static double _hoc_sigmoidal(void * _vptr);
    static double _hoc_sinusoidal(void * _vptr);


    /* connect user functions to hoc names */
    static VoidFunc hoc_intfunc[] = {
        {0, 0}
    };
    static Member_func _member_func[] = {
        {"loc", _hoc_loc_pnt},
        {"has_loc", _hoc_has_loc},
        {"get_loc", _hoc_get_loc_pnt},
        {"sigmoidal", _hoc_sigmoidal},
        {"sinusoidal", _hoc_sinusoidal},
        {"quadratic", _hoc_quadratic},
        {nullptr, nullptr}
    };
    static double _hoc_sigmoidal(void * _vptr) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        auto* const _pnt = static_cast<Point_process*>(_vptr);
        auto* const _p = _pnt->prop;
        if (!_p) {
            hoc_execerror("POINT_PROCESS data instance not valid", nullptr);
        }
        _nrn_mechanism_cache_instance _lmc{_p};
        size_t const id{};
        _ppvar = _nrn_mechanism_access_dparam(_p);
        _thread = _extcall_thread.data();
        nt = static_cast<NrnThread*>(_pnt->_vnt);
        auto inst = make_instance_tbl_point_process(_lmc);
        auto node_data = make_node_data_tbl_point_process(_p);
        update_table_sigmoidal_tbl_point_process(_lmc, inst, node_data, id, _ppvar, _thread, nt);
        _r = 1.;
        sigmoidal_tbl_point_process(_lmc, inst, node_data, id, _ppvar, _thread, nt, *getarg(1));
        return(_r);
    }
    static double _hoc_sinusoidal(void * _vptr) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        auto* const _pnt = static_cast<Point_process*>(_vptr);
        auto* const _p = _pnt->prop;
        if (!_p) {
            hoc_execerror("POINT_PROCESS data instance not valid", nullptr);
        }
        _nrn_mechanism_cache_instance _lmc{_p};
        size_t const id{};
        _ppvar = _nrn_mechanism_access_dparam(_p);
        _thread = _extcall_thread.data();
        nt = static_cast<NrnThread*>(_pnt->_vnt);
        auto inst = make_instance_tbl_point_process(_lmc);
        auto node_data = make_node_data_tbl_point_process(_p);
        update_table_sinusoidal_tbl_point_process(_lmc, inst, node_data, id, _ppvar, _thread, nt);
        _r = 1.;
        sinusoidal_tbl_point_process(_lmc, inst, node_data, id, _ppvar, _thread, nt, *getarg(1));
        return(_r);
    }
    static double _hoc_quadratic(void * _vptr) {
        double _r{};
        Datum* _ppvar;
        Datum* _thread;
        NrnThread* nt;
        auto* const _pnt = static_cast<Point_process*>(_vptr);
        auto* const _p = _pnt->prop;
        if (!_p) {
            hoc_execerror("POINT_PROCESS data instance not valid", nullptr);
        }
        _nrn_mechanism_cache_instance _lmc{_p};
        size_t const id{};
        _ppvar = _nrn_mechanism_access_dparam(_p);
        _thread = _extcall_thread.data();
        nt = static_cast<NrnThread*>(_pnt->_vnt);
        auto inst = make_instance_tbl_point_process(_lmc);
        auto node_data = make_node_data_tbl_point_process(_p);
        update_table_quadratic_tbl_point_process(_lmc, inst, node_data, id, _ppvar, _thread, nt);
        _r = quadratic_tbl_point_process(_lmc, inst, node_data, id, _ppvar, _thread, nt, *getarg(1));
        return(_r);
    }


    inline static int f_sigmoidal_tbl_point_process(_nrn_mechanism_cache_range& _lmc, tbl_point_process_Instance& inst, tbl_point_process_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _lv) {
        int ret_f_sigmoidal = 0;
        auto v = node_data.node_voltages[node_data.nodeindices[id]];
        inst.sig[id] = 1.0 / (1.0 + exp(inst.global->k * (_lv - inst.global->d)));
        return ret_f_sigmoidal;
    }


    void update_table_sigmoidal_tbl_point_process(_nrn_mechanism_cache_range& _lmc, tbl_point_process_Instance& inst, tbl_point_process_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt) {
        if (inst.global->usetable == 0) {
            return;
        }
        static bool make_table = true;
        static double save_k;
        static double save_d;
        if (save_k != inst.global->k) {
            make_table = true;
        }
        if (save_d != inst.global->d) {
            make_table = true;
        }
        if (make_table) {
            make_table = false;
            inst.global->tmin_sigmoidal =  -127.0;
            double tmax = 128.0;
            double dx = (tmax-inst.global->tmin_sigmoidal) / 155.;
            inst.global->mfac_sigmoidal = 1./dx;
            double x = inst.global->tmin_sigmoidal;
            for (std::size_t i = 0; i < 156; x += dx, i++) {
                f_sigmoidal_tbl_point_process(_lmc, inst, node_data, id, _ppvar, _thread, nt, x);
                inst.global->t_sig[i] = inst.sig[id];
            }
            save_k = inst.global->k;
            save_d = inst.global->d;
        }
    }


    inline static int sigmoidal_tbl_point_process(_nrn_mechanism_cache_range& _lmc, tbl_point_process_Instance& inst, tbl_point_process_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _lv){
        if (inst.global->usetable == 0) {
            f_sigmoidal_tbl_point_process(_lmc, inst, node_data, id, _ppvar, _thread, nt, _lv);
            return 0;
        }
        double xi = inst.global->mfac_sigmoidal * (_lv - inst.global->tmin_sigmoidal);
        if (isnan(xi)) {
            inst.sig[id] = xi;
            return 0;
        }
        if (xi <= 0. || xi >= 155.) {
            int index = (xi <= 0.) ? 0 : 155;
            inst.sig[id] = inst.global->t_sig[index];
            return 0;
        }
        int i = int(xi);
        double theta = xi - double(i);
        inst.sig[id] = inst.global->t_sig[i] + theta*(inst.global->t_sig[i+1]-inst.global->t_sig[i]);
        return 0;
    }


    inline static int f_sinusoidal_tbl_point_process(_nrn_mechanism_cache_range& _lmc, tbl_point_process_Instance& inst, tbl_point_process_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _lx) {
        int ret_f_sinusoidal = 0;
        auto v = node_data.node_voltages[node_data.nodeindices[id]];
        inst.v1[id] = sin(inst.global->c1 * _lx) + 2.0;
        inst.v2[id] = cos(inst.global->c2 * _lx) + 2.0;
        return ret_f_sinusoidal;
    }


    void update_table_sinusoidal_tbl_point_process(_nrn_mechanism_cache_range& _lmc, tbl_point_process_Instance& inst, tbl_point_process_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt) {
        if (inst.global->usetable == 0) {
            return;
        }
        static bool make_table = true;
        static double save_c1;
        static double save_c2;
        if (save_c1 != inst.global->c1) {
            make_table = true;
        }
        if (save_c2 != inst.global->c2) {
            make_table = true;
        }
        if (make_table) {
            make_table = false;
            inst.global->tmin_sinusoidal =  -4.0;
            double tmax = 6.0;
            double dx = (tmax-inst.global->tmin_sinusoidal) / 800.;
            inst.global->mfac_sinusoidal = 1./dx;
            double x = inst.global->tmin_sinusoidal;
            for (std::size_t i = 0; i < 801; x += dx, i++) {
                f_sinusoidal_tbl_point_process(_lmc, inst, node_data, id, _ppvar, _thread, nt, x);
                inst.global->t_v1[i] = inst.v1[id];
                inst.global->t_v2[i] = inst.v2[id];
            }
            save_c1 = inst.global->c1;
            save_c2 = inst.global->c2;
        }
    }


    inline static int sinusoidal_tbl_point_process(_nrn_mechanism_cache_range& _lmc, tbl_point_process_Instance& inst, tbl_point_process_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _lx){
        if (inst.global->usetable == 0) {
            f_sinusoidal_tbl_point_process(_lmc, inst, node_data, id, _ppvar, _thread, nt, _lx);
            return 0;
        }
        double xi = inst.global->mfac_sinusoidal * (_lx - inst.global->tmin_sinusoidal);
        if (isnan(xi)) {
            inst.v1[id] = xi;
            inst.v2[id] = xi;
            return 0;
        }
        if (xi <= 0. || xi >= 800.) {
            int index = (xi <= 0.) ? 0 : 800;
            inst.v1[id] = inst.global->t_v1[index];
            inst.v2[id] = inst.global->t_v2[index];
            return 0;
        }
        int i = int(xi);
        double theta = xi - double(i);
        inst.v1[id] = inst.global->t_v1[i] + theta*(inst.global->t_v1[i+1]-inst.global->t_v1[i]);
        inst.v2[id] = inst.global->t_v2[i] + theta*(inst.global->t_v2[i+1]-inst.global->t_v2[i]);
        return 0;
    }


    inline static double f_quadratic_tbl_point_process(_nrn_mechanism_cache_range& _lmc, tbl_point_process_Instance& inst, tbl_point_process_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _lx) {
        double ret_f_quadratic = 0.0;
        auto v = node_data.node_voltages[node_data.nodeindices[id]];
        ret_f_quadratic = inst.global->c1 * _lx * _lx + inst.global->c2;
        return ret_f_quadratic;
    }


    void update_table_quadratic_tbl_point_process(_nrn_mechanism_cache_range& _lmc, tbl_point_process_Instance& inst, tbl_point_process_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt) {
        if (inst.global->usetable == 0) {
            return;
        }
        static bool make_table = true;
        static double save_c1;
        static double save_c2;
        if (save_c1 != inst.global->c1) {
            make_table = true;
        }
        if (save_c2 != inst.global->c2) {
            make_table = true;
        }
        if (make_table) {
            make_table = false;
            inst.global->tmin_quadratic =  -3.0;
            double tmax = 5.0;
            double dx = (tmax-inst.global->tmin_quadratic) / 500.;
            inst.global->mfac_quadratic = 1./dx;
            double x = inst.global->tmin_quadratic;
            for (std::size_t i = 0; i < 501; x += dx, i++) {
                inst.global->t_quadratic[i] = f_quadratic_tbl_point_process(_lmc, inst, node_data, id, _ppvar, _thread, nt, x);
            }
            save_c1 = inst.global->c1;
            save_c2 = inst.global->c2;
        }
    }


    inline static double quadratic_tbl_point_process(_nrn_mechanism_cache_range& _lmc, tbl_point_process_Instance& inst, tbl_point_process_NodeData& node_data, size_t id, Datum* _ppvar, Datum* _thread, NrnThread* nt, double _lx){
        if (inst.global->usetable == 0) {
            return f_quadratic_tbl_point_process(_lmc, inst, node_data, id, _ppvar, _thread, nt, _lx);
        }
        double xi = inst.global->mfac_quadratic * (_lx - inst.global->tmin_quadratic);
        if (isnan(xi)) {
            return xi;
        }
        if (xi <= 0. || xi >= 500.) {
            int index = (xi <= 0.) ? 0 : 500;
            return inst.global->t_quadratic[index];
        }
        int i = int(xi);
        double theta = xi - double(i);
        return inst.global->t_quadratic[i] + theta * (inst.global->t_quadratic[i+1] - inst.global->t_quadratic[i]);
    }


    static void nrn_init_tbl_point_process(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_tbl_point_process(_lmc);
        auto node_data = make_node_data_tbl_point_process(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
            auto* _ppvar = _ml_arg->pdata[id];
            int node_id = node_data.nodeindices[id];
            auto v = node_data.node_voltages[node_id];
        }
    }


    static inline double nrn_current_tbl_point_process(_nrn_mechanism_cache_range& _lmc, NrnThread* nt, Datum* _ppvar, Datum* _thread, size_t id, tbl_point_process_Instance& inst, tbl_point_process_NodeData& node_data, double v) {
        double current = 0.0;
        sigmoidal_tbl_point_process(_lmc, inst, node_data, id, _ppvar, _thread, nt, v);
        inst.g[id] = 0.001 * inst.sig[id];
        inst.i[id] = inst.g[id] * (v - 30.0);
        current += inst.i[id];
        return current;
    }


    /** update current */
    static void nrn_cur_tbl_point_process(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_tbl_point_process(_lmc);
        auto node_data = make_node_data_tbl_point_process(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_data.nodeindices[id];
            double v = node_data.node_voltages[node_id];
            auto* _ppvar = _ml_arg->pdata[id];
            double I1 = nrn_current_tbl_point_process(_lmc, nt, _ppvar, _thread, id, inst, node_data, v+0.001);
            double I0 = nrn_current_tbl_point_process(_lmc, nt, _ppvar, _thread, id, inst, node_data, v);
            double rhs = I0;
            double g = (I1-I0)/0.001;
            double mfactor = 1.e2/(*inst.node_area[id]);
            g = g*mfactor;
            rhs = rhs*mfactor;
            node_data.node_rhs[node_id] -= rhs;
            inst.g_unused[id] = g;
        }
    }


    static void nrn_state_tbl_point_process(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_tbl_point_process(_lmc);
        auto node_data = make_node_data_tbl_point_process(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_data.nodeindices[id];
            auto* _ppvar = _ml_arg->pdata[id];
            auto v = node_data.node_voltages[node_id];
        }
    }


    static void nrn_jacob_tbl_point_process(const _nrn_model_sorted_token& _sorted_token, NrnThread* nt, Memb_list* _ml_arg, int _type) {
        _nrn_mechanism_cache_range _lmc{_sorted_token, *nt, *_ml_arg, _ml_arg->type()};
        auto inst = make_instance_tbl_point_process(_lmc);
        auto node_data = make_node_data_tbl_point_process(*nt, *_ml_arg);
        auto* _thread = _ml_arg->_thread;
        auto nodecount = _ml_arg->nodecount;
        for (int id = 0; id < nodecount; id++) {
            int node_id = node_data.nodeindices[id];
            node_data.node_diagonal[node_id] += inst.g_unused[id];
        }
    }
    static void nrn_destructor_tbl_point_process(Prop* prop) {
        Datum* _ppvar = _nrn_mechanism_access_dparam(prop);
        _nrn_mechanism_cache_instance _lmc{prop};
        const size_t id = 0;
        auto inst = make_instance_tbl_point_process(_lmc);
        auto node_data = make_node_data_tbl_point_process(prop);

    }


    static void _initlists() {
    }


    /** register channel with the simulator */
    extern "C" void _table_point_process_reg() {
        _initlists();

        _pointtype = point_register_mech(mechanism_info, nrn_alloc_tbl_point_process, nrn_cur_tbl_point_process, nrn_jacob_tbl_point_process, nrn_state_tbl_point_process, nrn_init_tbl_point_process, -1, 1, _hoc_create_pnt, _hoc_destroy_pnt, _member_func);

        mech_type = nrn_get_mechtype(mechanism_info[1]);
        hoc_register_parm_default(mech_type, &_parameter_defaults);
        _nrn_thread_table_reg(mech_type, _check_table_thread);
        _nrn_mechanism_register_data_fields(mech_type,
            _nrn_mechanism_field<double>{"g"} /* 0 */,
            _nrn_mechanism_field<double>{"i"} /* 1 */,
            _nrn_mechanism_field<double>{"v1"} /* 2 */,
            _nrn_mechanism_field<double>{"v2"} /* 3 */,
            _nrn_mechanism_field<double>{"sig"} /* 4 */,
            _nrn_mechanism_field<double>{"v_unused"} /* 5 */,
            _nrn_mechanism_field<double>{"g_unused"} /* 6 */,
            _nrn_mechanism_field<double*>{"node_area", "area"} /* 0 */,
            _nrn_mechanism_field<Point_process*>{"point_process", "pntproc"} /* 1 */
        );

        hoc_register_prop_size(mech_type, 7, 2);
        hoc_register_dparam_semantics(mech_type, 0, "area");
        hoc_register_dparam_semantics(mech_type, 1, "pntproc");
        hoc_register_var(hoc_scalar_double, hoc_vector_double, hoc_intfunc);
    }
}
