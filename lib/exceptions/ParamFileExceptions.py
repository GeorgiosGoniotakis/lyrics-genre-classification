class NoParamFileFound(Exception):
    def __init__(self, filename):
        super(str, "Error no parameter file found in path: {}.".format(filename))


class NoCategoryFound(Exception):
    def __init__(self, category):
        super(str, "No category with name {}, found in parameter file.".format(category))


class NoItemFound(Exception):
    def __init__(self, item):
        super(str, "No item with name {}, found in parameter file.".format(item))
