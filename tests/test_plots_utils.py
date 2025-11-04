from pymarlzooplus.utils.plot_utils import plot_single_experiment_results, plot_multiple_experiment_results, \
    plot_average_per_algo_for_all_tasks_of_a_benchmark

"""
This script contains examples of how to use the plotting utilities from pymarlzooplus.
TODO: Integrate these examples into unit tests.
"""

if __name__ == '__main__':

    ### Examples of plotting
    ## Single algo
    path_to_results_ = "~/sacred/emc/pistonball_v6/1"
    algo_name_ = "emc"
    env_name_ = "pistonball_v6"
    plot_single_experiment_results(path_to_results_, algo_name_, env_name_)

    ## Many algos
    paths_to_results_ = [
        "~/sacred/coma/pistonball_v6",
        "~/sacred/maa2c/pistonball_v6",
        "~/sacred/mappo/pistonball_v6",
        "~/sacred/qmix/pistonball_v6",
        "~/sacred/eoi/pistonball_v6",
        "~/sacred/qplex/pistonball_v6",
        "~/sacred/maser/pistonball_v6",
        "~/sacred/cds/pistonball_v6",
        "~/sacred/mat_dec/pistonball_v6",
        "~/sacred/emc/pistonball_v6",
        "~/sacred/happo/pistonball_v6"
    ]
    algo_names_ = ["COMA", "MAA2C", "MAPPO", "QMIX", "EOI", "QPLEX", "MASER", "CDS", "MAT-DEC", "EMC", "HAPPO"]
    env_name_ = "Pistonball"
    path_to_save_ = "~/multiple-exps-plots/pistonball_v6/"

    plot_train_ = False
    plot_legend_bool_ = False
    plot_multiple_experiment_results(
        paths_to_results_,
        algo_names_,
        env_name_,
        path_to_save_,
        plot_train_,
        plot_legend_bool_
    )

    ## Average plots per algo for all tasks of a benchmark
    _paths_to_pickle_results = [
        "~/multiple-exps-plots/pistonball_v6/all_results_env=pistonball_v6.pkl",
        "~/multiple-exps-plots/cooperative_pong_v5/all_results_env=cooperative_pong_v5.pkl",
        "~/multiple-exps-plots/entombed_cooperative_v3/all_results_env=entombed_cooperative_v3.pkl"
    ]
    _plot_title = "PettingZoo"
    _path_to_save = "~/multiple-exps-plots/pettingzoo/"

    _plot_legend = False
    plot_average_per_algo_for_all_tasks_of_a_benchmark(
        _paths_to_pickle_results,
        _plot_title,
        _path_to_save,
        _plot_legend
    )
