class NoMLModelFound(Exception):
    def __init__(self, model_name):
        super(str, "Error no model with name: {} found in collection.".format(model_name))


class WrongParameters(Exception):
    def __init__(self, parameters):
        super(str, "Wrong parameters on model initialization: {}".format(parameters))
