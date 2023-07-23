import abc


class DatasetMaker(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def process(self, input_filepath, output_filepath):
        """Processes multiple files from the {input_filepath} directory and
        generates output to {output_filepath} directory"""
        raise NotImplementedError

