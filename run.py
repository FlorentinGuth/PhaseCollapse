import shlex
import subprocess


imagenet_path = "PATH/TO/IMAGENET"


for dataset in ["cifar", "imagenet"]:
    data = dict(cifar="--cifar10", imagenet=f"--data {imagenet_path}")[dataset]
    n_blocks = dict(cifar=8, imagenet=11)[dataset]
    epochs = dict(cifar=300, imagenet=150)[dataset]
    lr_adjust = dict(cifar=70, imagenet=45)[dataset]
    factorize_filters = dict(cifar=0, imagenet=1)[dataset]
    angles = ' '.join([dict(cifar='4', imagenet='8')[dataset]] * n_blocks)

    cmd = f"{data} --n-blocks {n_blocks} -p 100 --scattering-wph {' '.join(['1'] * n_blocks)}" \
          f" --epochs {epochs} --learning-rate-adjust-frequency {lr_adjust} --lr 0.01 -j 10" \
          f" --weight-decay 0.0001 --batch-size 128 --factorize-filters {factorize_filters}" \
          f" --scat-angles {angles}"

    proj_sizes = []
    if dataset == "imagenet":
        proj_sizes.extend(["32", "64"])
    proj_sizes.extend(["64", "128", "256", "512", "512", "512", "512", "512"])
    if dataset == "imagenet":
        proj_sizes.append("256")
    Pc_sizes = f"--Pc-size {' '.join(proj_sizes)}"

    for skip in [False, True]:
        if skip:
            arch = "-a 'Fw Std Pc N' --psi-arch mod"
        else:
            arch = "-a 'Fw rho Std Pc N'"

        nonlins = dict(mod=[])
        # Add experiments for Table 2.
        if dataset == "cifar":
            nonlins.update(cst=[("bias", 0.1)], ms=[("gain", 1.0), ("bias", 0.0)], mc=[], tanh=[])

        for nonlin, params in nonlins.items():
            learned = ['0' if nonlin == "mod" else '1'] * n_blocks
            non_lin_str = f"--scat-non-linearity {' '.join([nonlin] * n_blocks)} --scat-non-linearity-learned {' '.join(learned)}"
            for param, value in params:
                values = [str(value)] * n_blocks
                non_lin_str = non_lin_str + f" --scat-non-linearity-{param} {' '.join(values)}"

            name = f"{dataset}-{nonlin}-skip-{skip}"
            to_run = f"python main_block.py {cmd} {arch} {Pc_sizes} {non_lin_str} --dir {name}"
            # subprocess.run(shlex.split(to_run), check=True, capture_output=True)


# Experiments for Table 3.
for non_linearity in ["relu", "abs", "thresh", "tanh", "powfixed"]:
    if non_linearity == "thresh":
        non_linearity_args = " --init-bias 1"
    else:
        non_linearity_args = ""


    name = f"resnet18-nobias-{non_linearity}"
    nonlin = f"--non-linearity {non_linearity}{non_linearity_args}"
    to_run = f"python standard_arch.py --arch resnet18-custom --no-bias {nonlin} --dir {name} {imagenet_path}"
    print(to_run)
    # subprocess.run(shlex.split(to_run), check=True, capture_output=True)
