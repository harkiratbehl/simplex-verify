import os

"""
Run the experiments to tune competing optimizers for the planet problem w/ variable splitting
"""

def run_incomplete_verif_nets(gpu_id, cpus, iters, define_linear_approximation, images, adam_algorithms, prox_algorithms, cut_algorithms, data):

    define_linear_approximation_string = "--define_linear_approximation" if define_linear_approximation else ""

    for img in images:
        for algo, beta1, inlr, finlr in adam_algorithms:

            # have adam run for more as it's faster
            adam_iters = int(iters * 2.6) if algo == "dj-adam" else int(iters * 1.85)

            command = f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python tools/bounding_tools/bounding_runner.py " \
                f"--eps 0.04705882352 --algorithm {algo} --out_iters {adam_iters} --img_idx" \
                f" {img} --init_step {inlr} --fin_step {finlr} --beta1 {beta1} {define_linear_approximation_string} " \
                f" --data {data} --network_filename ./data/cifar_madry_8px.pth"
            print(command)
            os.system(command)

        for algo, momentum, ineta, fineta in prox_algorithms:
            command = f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python tools/bounding_tools/bounding_runner.py " \
                f"--eps 0.04705882352 --algorithm {algo} --out_iters {iters} " \
                f"--img_idx {img} --eta {ineta} --feta {fineta} --prox_momentum {momentum} " \
                f"{define_linear_approximation_string} --data {data} " \
                f"--network_filename ./data/cifar_madry_8px.pth" 
            print(command)
            os.system(command)


def run_complete_verif_nets(gpu_id, cpus, iters, define_linear_approximation, net, pdprops, prop_idxs, adam_algorithms, prox_algorithms, cut_algorithms, data):

    define_linear_approximation_string = "--define_linear_approximation" if define_linear_approximation else ""
    pdprops_str = f"--pdprops {pdprops}" if pdprops else ""

    for prop_idx in prop_idxs:

        for algo, inlr, finlr, init_out_iters, cut_frequency, max_cuts, cut_add, init_init_step, init_fin_step in cut_algorithms:
            # have adam run for more as it's faster

            command = f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python tools/bounding_tools/bounding_runner.py " \
                      f"--nn_name {net} --prop_idx {prop_idx} {pdprops_str} --algorithm {algo} " \
                      f"--out_iters {iters} --data {data}" \
                      f" --init_out_iters {init_out_iters} --cut_frequency {cut_frequency} --cut_add {cut_add}  --max_cuts {max_cuts}" \
                      f" --init_init_step {init_init_step} --init_fin_step {init_fin_step}" \
                      f" --init_step {inlr} --fin_step {finlr} {define_linear_approximation_string}"
            print(command)
            os.system(command)

        for algo, beta1, inlr, finlr in adam_algorithms:
            # have adam run for more as it's faster
            adam_iters = int(iters * 2.6) if algo == "dj-adam" else int(iters * 1.6)

            command = f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python tools/bounding_tools/bounding_runner.py " \
                      f"--nn_name {net} --prop_idx {prop_idx} {pdprops_str} --algorithm {algo} " \
                      f"--out_iters {adam_iters} --data {data}" \
                      f" --init_step {inlr} --fin_step {finlr} --beta1 {beta1} {define_linear_approximation_string}"
            print(command)
            os.system(command)

        for algo, momentum, ineta, fineta in prox_algorithms:
            command = f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python tools/bounding_tools/bounding_runner.py " \
                f"--nn_name {net} --prop_idx {prop_idx} {pdprops_str} --algorithm {algo} --out_iters {iters} " \
                f"--eta {ineta} --feta {fineta} --prox_momentum {momentum} {define_linear_approximation_string} --data {data}"
            print(command)
            os.system(command)


if __name__ == "__main__":

    gpu_id = 0  # 0
    cpus = "0-5"  # "10-14"

    iters = 100#500#1000#2000#4000

    define_linear_approximation = False

    # images = [0, 5, 10, 15, 20]
    cut_algorithms = [
        # algo, inlr, finlr, init_out_iters, cut_frequency, max_cuts, cut_add, init_init_step, init_fin_step
        # ("cut", 1e-3, 1e-6, 100, 450, 3, 10, 1e-2, 1e-4),
        # ("cut", 1e-3, 1e-6, 100, 450, 3, 10, 1e-1, 1e-3),
        # ("cut", 1e-3, 1e-6, 100, 450, 3, 10, 1e-4, 1e-6),
        # ("cut", 1e-3, 1e-4, 500, 450, 3, 10, 1e-2, 1e-4),
        # ("cut", 1e-3, 1e-4, 500, 450, 3, 10, 1e-1, 1e-3),
        # ("cut", 1e-3, 1e-4, 500, 450, 3, 10, 1e-4, 1e-6),
        # ("cut", 1e-3, 1e-6, 500, 450, 3, 10, 1e-2, 1e-4),
        # ("cut", 1e-3, 1e-6, 500, 450, 3, 10, 1e-1, 1e-3),
        # ("cut", 1e-3, 1e-6, 500, 450, 3, 10, 1e-1, 5e-2),
        # ("cut", 1e-3, 1e-6, 500, 450, 3, 10, 1e-1, 5e-3),
        # ("cut", 1e-3, 1e-6, 500, 450, 3, 10, 1e-1, 1e-4),
        # ("cut", 1e-3, 1e-6, 500, 450, 3, 10, 1e-4, 1e-6),
        # ("cut", 1e-3, 1e-6, 800, 450, 3, 10, 1e-1, 1e-3),
        # ("cut", 1e-3, 1e-6, 600, 450, 3, 10, 1e-1, 1e-3),
        # ("cut", 1e-3, 1e-6, 900, 450, 3, 10, 1e-1, 1e-3),
        # ("cut", 1e-3, 1e-6, 700, 450, 3, 10, 1e-1, 1e-3),
        # ("cut", 1e-3, 1e-6, 1000, 450, 3, 10, 1e-1, 1e-3),##best option for cifar10_8_255:100/1000 iters. cifar10_2_255:100/500 iters for mnist_0.3 and mnist_0.1: 100. 1000 iters is also very good
        # ("cut", 1e-3, 1e-6, 500, 450, 3, 10, 1e-1, 1e-3), ##best for mnist_convSmallRELU__Point, cifar10_convSmallRELU__Point
        # ("cut", 1e-3, 1e-5, 500, 450, 3, 10, 1e-1, 1e-3), 
        # ("cut", 1e-4, 1e-5, 500, 450, 3, 10, 1e-1, 1e-3), 
        # ("cut", 1e-4, 1e-6, 500, 450, 3, 10, 1e-1, 1e-3), 
        # ("cut", 1e-2, 1e-4, 500, 450, 3, 10, 1e-1, 1e-3), 
        # ("cut", 1e-2, 1e-5, 500, 450, 3, 10, 1e-1, 1e-3), 
        # ("cut", 1e-3, 1e-6, 500, 450, 3, 10, 1e-1, 1e-4), 
        # ("cut", 1e-3, 1e-6, 500, 450, 3, 10, 1e-1, 1e-5), 
        # ("cut", 1e-3, 1e-6, 500, 450, 3, 10, 1e-2, 1e-3), 
        # ("cut", 1e-3, 1e-6, 500, 450, 3, 10, 1e-2, 1e-4), 
        # ("cut", 1e-3, 1e-6, 500, 450, 3, 10, 1e-2, 1e-5), 
        # ("cut", 1e-3, 1e-6, 500, 450, 3, 10, 1e-3, 1e-4), 
        # ("cut", 1e-3, 1e-6, 500, 450, 3, 10, 1e-3, 1e-5), 
        # ("cut", 1e-3, 1e-6, 500, 450, 3, 10, 1e-3, 1e-6), 
        # ("cut", 1e-2, 1e-4, 500, 450, 3, 10, 1e-1, 1e-3), 
        ##("cut", 1e-2, 1e-4, 1000, 450, 3, 10, 1e-1, 1e-3),
        ##("cut", 1e-4, 1e-6, 1000, 450, 3, 10, 1e-1, 1e-3),
        ##("cut", 1e-3, 1e-5, 1000, 450, 3, 10, 1e-1, 1e-3),
        # ("cut", 1e-3, 1e-6, 2000, 450, 3, 10, 1e-1, 1e-3),
        # ("cut", 1e-3, 1e-6, 4000, 450, 3, 10, 1e-1, 1e-3),
        # ("cut", 1e-3, 1e-6, 1000, 450, 3, 10, 1e-2, 1e-4),
        # ("cut", 1e-3, 1e-6, 1000, 450, 3, 10, 1e-4, 1e-6),
        # ("cut", 1e-2, 1e-3, 500, 450, 3, 10, 1e-2, 1e-4),
        # ("cut", 1e-2, 1e-3, 500, 450, 3, 10, 1e-1, 1e-3),
        # ("cut", 1e-2, 1e-4, 500, 450, 3, 10, 1e-2, 1e-4),
        # ("cut", 1e-2, 1e-4, 500, 450, 3, 10, 1e-1, 1e-3),
        # ("cut", 1e-2, 1e-5, 500, 450, 3, 10, 1e-2, 1e-4),
        # ("cut", 1e-2, 1e-5, 500, 450, 3, 10, 1e-1, 1e-3),
        # ("cut", 1e-2, 1e-6, 500, 450, 3, 10, 1e-2, 1e-4),
        # ("cut", 1e-2, 1e-6, 500, 450, 3, 10, 1e-1, 1e-3),
        # ("cut", 1e-4, 1e-5, 500, 450, 3, 10, 1e-2, 1e-4),
        # ("cut", 1e-4, 1e-5, 500, 450, 3, 10, 1e-1, 1e-3),
        # ("cut", 1e-4, 1e-6, 500, 450, 3, 10, 1e-2, 1e-4),
        # ("cut", 1e-4, 1e-6, 500, 450, 3, 10, 1e-1, 1e-3),

    ]
    adam_algorithms = [
        # algo, beta1, inlr, finlr
        # ("planet-adam", 0.9, 1e-2, 1e-4),
        # ("planet-adam", 0.9, 1e-3, 1e-6),
        # ("planet-adam", 0.9, 1e-4, 1e-6),# Best for cifar10_8_255 and cifar10_2_255
        # ("planet-adam", 0.9, 1e-3, 1e-4),
        # ("planet-adam", 0.9, 1e-2, 1e-2),
        # ("planet-adam", 0.9, 1e-3, 1e-3),
        # ("planet-adam", 0.9, 1e-4, 1e-4),
        # ("planet-adam", 0.9, 1e-1, 1e-4),
        # ("planet-adam", 0.9, 1e-1, 1e-3),  # Best for mnist
    ]
    prox_algorithms = [  # momentum tuning
        # algo, momentum, ineta, fineta
        # ("proxlp", 0.0, 1e2, 1e2),  # Best for mnist, mnist_convSmallRELU__Point
        # ("proxlp", 0.0, 1e1, 1e1), # Best for cifar10_8_255 and cifar10_2_255
        # ("proxlp", 0.0, 1e3, 1e3),
        # ("proxlp", 0.0, 1e0, 1e0),  # Best for cifar10_2_255 and cifar10_8_255
        # ("proxlp", 0.0, 1e4, 1e4),
        # ("proxlp", 0.0, 5e1, 1e2),
        # ("proxlp", 0.0, 5e2, 1e3),
        # ("proxlp", 0.0, 5e0, 1e1),
        # ("proxlp", 0.0, 5e3, 1e4),
    ]

    # Incomplete verification experiments (UAI net)
    # run_incomplete_verif_nets(gpu_id, cpus, iters, define_linear_approximation, images, adam_algorithms, prox_algorithms, data, net)

    # Complete verification experiments (BaB nets)
    # data = "cifar"
    # net = "cifar_base_kw"
    # pdprops = "jodie-base_hard.pkl"

    ## run_incomplete_verif_nets function to be used for mnist-eth
    # data = "mnist"
    # nets = ["mnist_0.1", "mnist_0.3"]  # mnist_0.1 or mnist_0.3
    # pdprops = None
    # images = list(range(100)) # all 100 images for workshop
    # # read images from file
    # for net in nets:
    #     if net == 'mnist_0.1':
    #         images = [18, 92]
    #     elif net == 'mnist_0.3':
    #         images = [7, 19, 44, 58, 68, 69, 78, 80, 93, 95]
    #     run_complete_verif_nets(gpu_id, cpus, iters, define_linear_approximation, net, pdprops, images, adam_algorithms, prox_algorithms, cut_algorithms, data)

    # cifar-eth
    # data = "cifar10"
    # net = "cifar10_8_255"  # cifar10_8_255 or cifar10_2_255
    # pdprops = None
    # if data == 'cifar10':
    #     # with open(f"./data/correct-{net}.txt", "r") as file:
    #     #     line = file.readline()
    #     # images = line.split(", ")[:-1]
    #     # images = [int(img) for img in images]
    #     if net == 'cifar10_8_255':
    #         images = [4,38,49,55,88,96]#[4, 38, 49, 55, 96]
    #         images = [2,5,10,14,15,28,45,51,64,66,71,72,73,76,80,84,97]
    #     elif net == 'cifar10_2_255':
    #         images = [5,39,77,93]
    #     run_complete_verif_nets(gpu_id, cpus, iters, define_linear_approximation, net, pdprops, images, adam_algorithms, prox_algorithms, cut_algorithms, data)

    # eran
    # data = "mnist"#cifar10
    # net = "mnist_convSmallRELU__Point"  # mnist_convSmallRELU__Point, mnist_convSmallRELU__DiffAI, mnist_convSmallRELU__PGDK
    # images=[4,7,21,26,35,40,44,58,59,66,84]

    # data = "cifar10"
    # net = "cifar10_convSmallRELU__Point"  # cifar10_convSmallRELU__Point, cifar10_convSmallRELU__DiffAI, cifar10_convSmallRELU__PGDK
    images=[12,14,23,28,34,40,44,48,54,77,92]

    pdprops = None
    # images = list(range(100)) # all 100 images for workshop
    run_complete_verif_nets(gpu_id, cpus, iters, define_linear_approximation, net, pdprops, images, adam_algorithms, prox_algorithms, cut_algorithms, data)

    
