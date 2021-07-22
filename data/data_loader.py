def get_data_loader(opt, generator, directory):
    return generator.flow_from_directory(
        directory=directory,
        target_size=(opt.target_size_x, opt.target_size_y),
        batch_size=opt.batch_size,
        shuffle=opt.shuffle,
        seed=opt.seed,
        interpolation=opt.interpolation,
    )
