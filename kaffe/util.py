from tensorflow import Network

import inspect
import imp


def get_network_class_from_module(path_to_module):
    module = imp.load_source("module", path_to_module)
    members = inspect.getmembers(module)
    members = filter(lambda member: type(member[1]) == type, members)
    members = filter(lambda member: member[1] != Network, members)
    assert len(members) == 1, "Module should not contain another class definitions."
    Net = members[0][1]
    assert issubclass(Net, Network), "Class must be derived from kaffe.tensorflow.Network."
    return Net
