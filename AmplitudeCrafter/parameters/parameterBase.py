import re
from abc import ABC, abstractmethod
from warnings import warn
from jitter.fitting import FitParameter


def find_next(string: str, key: str):
    words = string.split()
    if key not in words:
        return string, None
    index = words.index(key)
    if index + 1 >= len(words):
        raise ValueError(f"Found Key {key}, but no value was supplied: {string}!")
    words_filtered = [w for i, w in enumerate(words) if i != index and i != index + 1]
    return " ".join(words_filtered), try_float(words[index + 1])


def check_const(string: str):
    if "const" in string:
        return string.replace("const", ""), True
    return string, False


def check_float(string: str):
    try:
        float(string)
        return True
    except ValueError:
        return False


def fail_false(func):
    def inner(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return False
    return inner


def find_parenthesis(string: str):
    opening = [i for i, c in enumerate(string) if c == "("]
    closing = [i for i, c in enumerate(string) if c == ")"]
    return opening, closing


def closing_index(string: str, opening_index):
    opening, closing = find_parenthesis(string)
    if opening_index not in opening:
        raise ValueError(f"Index {opening_index} not in index list for opening parenthesis {opening}!s")

    opening = [o for o in opening if o >= opening_index]
    closing = [c for c in closing if c >= opening_index]

    n_open = [len([o for o in opening if o < c]) - i - 1 for i, c in enumerate(closing)]

    return closing[n_open.index(0)]


def inside(opening, string: str):
    if not "(" in opening:
        raise ValueError("")

    opening_index = string.index(opening) + len(opening) - 1
    closing_ind = closing_index(string, opening_index=opening_index)

    return string[opening_index + 1: closing_ind], opening_index, closing_ind


def find_if_named(name, value):
    if not isinstance(value, str):
        return name, value, False

    def outside_brackets(string, index):
        return sum(1 if c == "(" else -1 if c == ")" else 0 for i, c in enumerate(string[:index])) == 0

    pattern = r"NAMED\(([^)]+)\)"
    match = re.search(pattern, value)

    if match and outside_brackets(value, match.start()):
        new_name = match.group(1)
        new_value = re.sub(pattern, "", value)
        return new_name, new_value, True
    else:
        return name, value, False



def no_name_string(f):
    def inner(cls, string, *args):
        name, value, _ = find_if_named("temporaryName", string)
        return f(cls, value)
    return inner


def append_name(f):
    def inner(param):
        name, _, named = find_if_named(param.name, param.initial_value)
        if named:
            return f(param) + f" NAMED({name})"
        return f(param) + ""
    return inner


def try_float(string: str):
    if not isinstance(string, str):
        raise ValueError(f"Only string allowed not {type(string)}!")
    try:
        return float(string)
    except ValueError:
        return string


class ParameterScope:
    def __init__(self, known_params=None):
        self.dict = known_params or {}

    def __enter__(self):
        self.dict_before = parameter.set_backend(self.dict)
        return self

    def __exit__(self, *args):
        parameter.set_backend(self.dict_before)


class parameter(ABC):
    parameters = dict()

    @classmethod
    @abstractmethod
    def match(cls, string):
        return False

    @abstractmethod
    def __call__(self):
        raise NotImplementedError()

    @abstractmethod
    def __init__(self, name: str):
        if self.check_exists():
            return False
        parameter.parameters[self.name] = self
        return True

    @property
    @abstractmethod
    def dict(self):
        raise NotImplementedError()

    def floating(self):
        return not self.const

    def check_exists(self):
        return hasattr(self, "name") and getattr(self, "name") in parameter.parameters

    @classmethod
    def clear(cls):
        cls.parameters = {}

    @classmethod
    def set_backend(cls, known_params: dict):
        dtc = cls.parameters
        cls.parameters = known_params
        return dtc

    def __new__(cls, name, value, *args, **kwargs):
        new_obj = object.__new__(cls)
        new_obj.initial_value = value
        name, value, named = find_if_named(name, value)
        new_obj.name = name
        new_obj.value_string = value
        return parameter.parameters.get(name, new_obj)
