from configparser import ConfigParser
import socket

class Configure(ConfigParser):
    config_filepath = "config.init"
    def __init__(self):
        defaults = {"LOG_PATH": {"model": r".\models",
                                 "run": r".\runs",
                                 "checkpoint": r".\checkpoints"},
                    "ID": {"computer": socket.gethostname(),},
                   }
        ConfigParser.__init__(self)
        self.read_dict(defaults)
        with open("config.ini", "r") as f:
            self.read(f)
        
    def write(self):
        with open("config.ini", "w") as f:
            ConfigParser.write(self, f)

config = Configure()