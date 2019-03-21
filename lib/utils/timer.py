import time


class Timer:
    """
    Implements a basic timer to track the overall
    duration of the execution of a script.
    """
    def __init__(self):
        """
        Initializes a timer and starts the clock.
        """
        self.__start_time = time.time()

    def get_time(self):
        """Used to retrieve the overall time.

        Returns:
            The overall time as a string
        """
        return time.time() - self.__start_time
