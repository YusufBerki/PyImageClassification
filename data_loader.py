from opt import DATA_LOADER_CONFIG


def get_data_loader(generator, directory):
    return generator.flow_from_directory(directory=directory)
