import os
import argparse

def run_incomplete_verification(gpu, cpus, experiment, epsilon):

    if experiment == "cifar_sgd":
        command = f"CUDA_VISIBLE_DEVICES={gpu} taskset -c {cpus} " \
                  f"python tools/bounding_tools/simplex_cifar_bound_comparison.py " \
                  f"--network_filename ./data/cifar_sgd_8px.pth --eps {epsilon} --target_directory ./results/sgd8_ll --from_intermediate_bounds"
    elif experiment == 'cifar_model_wide_l1':
        command = f"CUDA_VISIBLE_DEVICES={gpu} taskset -c {cpus} " \
                    f"python tools/bounding_tools/simplex_cifar_bound_comparison.py  --nn_name cifar_model_wide_l1 " \
                    f"--network_filename ./data/cifar_madry_8px.pth --eps {epsilon} --target_directory ./results/cifar_model_wide_l1 --from_intermediate_bounds"
    elif experiment == 'cifar_model_deep_l1':
        command = f"CUDA_VISIBLE_DEVICES={gpu} taskset -c {cpus} " \
                    f"python tools/bounding_tools/simplex_cifar_bound_comparison.py  --nn_name cifar_model_deep_l1 " \
                    f"--network_filename ./data/cifar_madry_8px.pth --eps {epsilon} --target_directory ./results/cifar_model_deep_l1 --from_intermediate_bounds"
    elif experiment == 'cifar_model_base_l1':
        command = f"CUDA_VISIBLE_DEVICES={gpu} taskset -c {cpus} " \
                    f"python tools/bounding_tools/simplex_cifar_bound_comparison.py  --nn_name cifar_model_base_l1 " \
                    f"--network_filename ./data/cifar_madry_8px.pth --eps {epsilon} --target_directory ./results/cifar_model_base_l1 --from_intermediate_bounds"
    elif experiment == "mnist_l1":
        command = f"CUDA_VISIBLE_DEVICES={gpu} taskset -c {cpus} " \
                  f"python tools/bounding_tools/simplex_mnist_bound_comparison.py " \
                  f"--network_filename ./data/cifar_sgd_8px.pth --eps 0.35 --target_directory ./results/mnist_l1_0.35 --from_intermediate_bounds"

    print(command)
    os.system(command)

def ver_acc_incomplete_verification(gpu, cpus, experiment, epsilon):

    if experiment == "cifar_sgd":
        command = f"CUDA_VISIBLE_DEVICES={gpu} taskset -c {cpus} " \
                  f"python tools/bounding_tools/simplex_cifar_ver_acc.py " \
                  f"--network_filename ./data/cifar_sgd_8px.pth --eps {epsilon} --target_directory ./results/sgd8_ll --from_intermediate_bounds"
    elif experiment == 'cifar_model_wide_l1':
        command = f"CUDA_VISIBLE_DEVICES={gpu} taskset -c {cpus} " \
                    f"python tools/bounding_tools/simplex_cifar_ver_acc.py  --nn_name cifar_model_wide_l1 " \
                    f"--network_filename ./data/cifar_madry_8px.pth --eps {epsilon} --target_directory ./results/cifar_model_wide_l1 --from_intermediate_bounds"
    elif experiment == 'cifar_model_deep_l1':
        command = f"CUDA_VISIBLE_DEVICES={gpu} taskset -c {cpus} " \
                    f"python tools/bounding_tools/simplex_cifar_ver_acc.py  --nn_name cifar_model_deep_l1 " \
                    f"--network_filename ./data/cifar_madry_8px.pth --eps {epsilon} --target_directory ./results/cifar_model_deep_l1 --from_intermediate_bounds"
    elif experiment == 'cifar_model_base_l1':
        command = f"CUDA_VISIBLE_DEVICES={gpu} taskset -c {cpus} " \
                    f"python tools/bounding_tools/simplex_cifar_ver_acc.py  --nn_name cifar_model_base_l1 " \
                    f"--network_filename ./data/cifar_madry_8px.pth --eps {epsilon} --target_directory ./results/cifar_model_base_l1 --from_intermediate_bounds"
    elif experiment == "mnist_l1":
        command = f"CUDA_VISIBLE_DEVICES={gpu} taskset -c {cpus} " \
                  f"python tools/bounding_tools/simplex_cifar_ver_acc.py " \
                  f"--network_filename ./data/cifar_sgd_8px.pth --eps 0.35 --target_directory ./results/mnist_l1_0.35 --from_intermediate_bounds"
    print(command)
    os.system(command)

if __name__ == "__main__":

    # Example: python scripts/run_anderson_incomplete.py --gpu_id 0 --cpus 0-3 --experiment cifar_sgd
    # Example: python scripts/run_anderson_incomplete.py --gpu_id 1 --cpus 5-8 --experiment cifar_madry

    # Note: the experiments take several days to terminate. For each of the three experiments, consider terminating the
    # experiment after roughly 1000 images have been completed.

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, help='Argument of CUDA_VISIBLE_DEVICES', default='0')
    parser.add_argument('--cpus', type=str, help='Argument of taskset -c, 4 cpus are required for Gurobi', default='1-4')
    parser.add_argument('--experiment', type=str, help='Which experiment to run', choices=["cifar_sgd", "cifar_madry", "all", "cifar_model_wide_l1", "cifar_model_deep_l1", "cifar_model_deeper_l1", "cifar_model_large_l1", "cifar_model_base_l1", "mnist_l1"])
    parser.add_argument('--epsilon_list', default=[0.5])
    args = parser.parse_args()

    # Computing bounds
    for experiment in ["cifar_sgd", "cifar_model_wide_l1", "cifar_model_deep_l1", "cifar_model_base_l1", "mnist_l1"]:
        for epsilon in args.epsilon_list:
            run_incomplete_verification(args.gpu_id, args.cpus, experiment, epsilon)
