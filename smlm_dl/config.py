from configparser import ConfigParser
import socket

class Configure(ConfigParser):
    config_filepath = "config.ini"
    def __init__(self):
        defaults = {"LOG_PATH": {"run": r".\runs",},
                    "ID": {"computer": socket.gethostname(),},
                    "TEST_DATASET_PATH": {"hagen": r".\datasets\hagen",
                                         "epfl": r".\datasets\epfl"},
                   }
        ConfigParser.__init__(self)
        self.read_dict(defaults)
        self.read(self.config_filepath)
        self.write()
        
    def write(self):
        with open(self.config_filepath, "w") as f:
            ConfigParser.write(self, f)

config = Configure()