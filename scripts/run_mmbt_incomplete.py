import os
import argparse

def run_incomplete_verification(gpu, cpus, experiment):

    if experiment == "mmbt_newer":
        command = f"CUDA_VISIBLE_DEVICES={gpu} taskset -c {cpus} " \
                  f"python tools/bounding_tools/simplex_mmbt_bound_comparison.py " \
                  f"--network_filename data/mmbt_save_newer_1_10_0.65pgd_10_0.26_0.3/mmbt_model_run/model_best.pt --target_directory ./results/mmbt_save_newer_1_10_0.65pgd_10_0.26_0.3 --from_intermediate_bounds"

    print(command)
    os.system(command)

if __name__ == "__main__":

    # Example: python scripts/run_anderson_incomplete.py --gpu_id 0 --cpus 0-3 --experiment cifar_sgd
    # Example: python scripts/run_anderson_incomplete.py --gpu_id 1 --cpus 5-8 --experiment cifar_madry

    # Note: the experiments take several days to terminate. For each of the three experiments, consider terminating the
    # experiment after roughly 1000 images have been completed.

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, help='Argument of CUDA_VISIBLE_DEVICES', default='0')
    parser.add_argument('--cpus', type=str, help='Argument of taskset -c, 4 cpus are required for Gurobi', default='8-10')
    parser.add_argument('--experiment', type=str, help='Which experiment to run', choices=["mmbt_newer"])
    args = parser.parse_args()

    for experiment in ["mmbt_newer"]:
        run_incomplete_verification(args.gpu_id, args.cpus, experiment)