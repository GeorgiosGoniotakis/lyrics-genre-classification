class GenreFileNotExists(Exception):
    def __init__(self, filepath):
        super(str, "File containing the genres does not exist on path: {}".format(filepath))
