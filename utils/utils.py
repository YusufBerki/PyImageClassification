import os


def makedirs(path):
    dir_name, _ = os.path.split(path)
    os.makedirs(dir_name, exist_ok=True)
    return path


def fix_opt_paths(opt):
    opt.train_dataset_dir = os.path.join(opt.dataset_path, opt.train_dataset_dir)
    opt.test_dataset_dir = os.path.join(opt.dataset_path, opt.test_dataset_dir)
    return opt


def get_suffix(opt, prefix):
    if not isinstance(opt, dict):
        opt = vars(opt)

    return {k.replace(prefix, ''): v for k, v in opt.items() if k.startswith(prefix) and k != prefix}
