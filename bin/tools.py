# code by Ruiyang

import logging
from collections import UserDict
from collections.abc import Container, Collection, Mapping
from contextlib import contextmanager
from typing import Any, Final

MISSING: Final = object()


def is_builtin_seq(x):
    return isinstance(x, (tuple, list))


def format_collection(c: Collection, nmax=5, sep=", ") -> str:
    if isinstance(c, str):
        return c
    import math
    import numpy as np

    type = c.__class__
    if type is list:
        sbegin, send = "[", "]"
    elif type is tuple:
        sbegin, send = "(", ")"
    elif type is set or type is dict:
        sbegin, send = "{", "}"
    elif type is frozenset:
        sbegin, send = "frozenset({", "})"
    elif type is np.ndarray:
        sbegin, send = "array([", "])"
    else:
        sbegin, send = f"{type.__name__}(", ")"

    if len(c) <= nmax:
        smid = sep.join(map(str, c))
    else:
        ileft = math.ceil(nmax / 2)
        iright = len(c) - (nmax - ileft)
        items: list[str] = []
        if isinstance(c, Mapping):
            for i, (k, v) in enumerate(c.items()):
                if i < ileft or i >= iright:
                    items.append(f"{k}: {v}")
        else:
            for i, x in enumerate(c):
                if i < ileft or i >= iright:
                    items.append(str(x))
        items.insert(ileft, "...")
        smid = sep.join(items)
    return f"{sbegin}{smid}{send}"


@contextmanager
def log_root_only(logroot=0, mpicomm=None):
    if mpicomm is None:
        from mpi4py import MPI

        mpicomm = MPI.COMM_WORLD
    if mpicomm.rank != logroot:
        logging.disable(logging.INFO)
    try:
        yield
    finally:
        if mpicomm.rank != logroot:
            logging.disable(logging.NOTSET)


def task_split(ntasks, rank, nprocs):
    subtask_size = ntasks // nprocs + (rank < ntasks % nprocs)
    istart = (ntasks // nprocs) * rank + (rank if rank < ntasks % nprocs else ntasks % nprocs)
    return slice(istart, istart + subtask_size)


class StaticMapping(Mapping):
    @classmethod
    def select(cls, tracers: Container[str] | None = None, return_dict=False) -> Any:
        all_tracers = {
            _: getattr(cls, _)
            for _ in dict.fromkeys(_ for _ in dir(cls) if not callable(getattr(cls, _)) and not _.startswith("_"))
        }
        if tracers is not None:
            all_tracers = {k: v for k, v in all_tracers.items() if k in tracers}
        return all_tracers if return_dict else all_tracers.values()

    @classmethod
    def keys(cls):
        return cls.select(return_dict=True).keys()

    @classmethod
    def values(cls):
        return cls.select(return_dict=True).values()

    @classmethod
    def items(cls):
        return cls.select(return_dict=True).items()

    def __repr__(self) -> str:
        all_tracers = list(self.select(return_dict=True).keys())
        return f"{type(self).__name__}(" + format_collection(all_tracers) + ")"

    def __getitem__(self, key):
        return getattr(self, key)

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())


class NestedDict(UserDict):
    """A dictionary supporting access to elements using `x.y.z`-style keys"""

    def __getitem__(self, key) -> Any:
        keys = key.split(".")
        v = self.data
        for i, key in enumerate(keys):
            if not isinstance(v, dict):
                raise TypeError(f"{'.'.join(keys[:i])} is not a dictionary")
            if key in v:
                v = v[key]
                continue
            if hasattr(self.__class__, "__missing__"):
                return self.__class__.__missing__(self, key)  # type: ignore
            raise KeyError(key)
        return v

    def __setitem__(self, key, item):
        keys = key.split(".")
        v = self.data
        entry, entry_name = None, None
        ilast = len(keys) - 1
        for i, key in enumerate(keys):
            if not isinstance(v, dict):
                if entry is not None:
                    del entry[entry_name]
                raise TypeError(f"{'.'.join(keys[:i])} is not a dictionary")
            if i != ilast:
                if key not in v:
                    v[key] = {}
                    if entry is None:
                        entry, entry_name = v, key
                v = v[key]
            else:
                v[key] = item

    def __delitem__(self, key):
        try:
            _ = self[key]
        except (KeyError, TypeError):
            raise
        keys = key.split(".")
        v = self.data
        for key in keys[:-1]:
            v = v[key]
        del v[keys[-1]]
