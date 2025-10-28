# code by Ruiyang

from __future__ import annotations
import numpy as np
from collections import UserDict
from desilike.samples import Chain
from pathlib import Path
from typing import Any, Literal


def add_ap_params(chain):
    if "qiso" not in chain and "qap" not in chain:
        if "qpar" not in chain or "qper" not in chain:
            return chain
        # qiso, qap => qpar, qper
        add_derived_param(
            chain,
            chain["qpar"] ** (1 / 3) * chain["qper"] ** (2 / 3),
            "qiso",
            latex=R"q_\mathrm{iso}",
        )
        add_derived_param(chain, chain["qpar"] / chain["qper"], "qap", latex=R"q_\mathrm{ap}")
    elif "qpar" not in chain and "qper" not in chain:
        if "qiso" not in chain or "qap" not in chain:
            return chain
        # qpar, qper => qiso, qap
        add_derived_param(chain, chain["qap"] ** (2 / 3) * chain["qiso"], "qpar", latex=R"q_\parallel")
        add_derived_param(chain, chain["qiso"] / chain["qap"] ** (1 / 3), "qper", latex=R"q_\perp")
    return chain


def df_to_dfs8(df, dm, a=0.6, rs_drag=99.07921342319206, R=8):
    return df * np.exp(dm / (2 * a) * np.tanh(a * np.log(rs_drag / R)))


def add_dfs8(chain, a=0.6, rs_drag=99.07921342319206, R=8):
    if "df" in chain and "dm" in chain and "dfs8" not in chain:
        return add_derived_param(
            chain,
            df_to_dfs8(df=chain["df"], dm=chain["dm"], a=a, rs_drag=rs_drag, R=R),
            "dfs8",
            latex=R"$f\sigma_{s8}/(f\sigma_{s8})_\mathrm{fid}$",
        )
    return chain


def add_derived_param(chain, array, name: str, latex: str | None = None, derived: bool | str = True):
    from desilike import Parameter, ParameterArray

    latex = latex or name
    chain[name] = ParameterArray(
        array, param=Parameter(name, derived=derived, latex=latex, drop=True, fixed=False)  # type: ignore
    )
    return chain


def load_bestfits(path, **kwargs):
    path = Path(path)
    if path.suffix == ".npy":
        from desilike.samples import Profiles

        profile = Profiles.load(path)
        retval = {}
        for k, v in profile.bestfit.choice(**kwargs).items():  # type: ignore
            try:
                v.item()
            except (AttributeError, ValueError):
                continue
            else:
                retval[k] = v.item()
        return retval
    elif path.suffix == ".minimum":
        # cobaya bestfits
        params_values_dict = {}
        with path.open("r") as f:
            skiprows = 3  # skip -log(Like) and chi-sq
            for i, line in enumerate(f):
                if i < skiprows:
                    continue
                if not line.strip():
                    if kwargs.get("input", False):
                        break
                    continue
                idx, value, name, *latex = line.strip().split()
                params_values_dict[name] = float(value)
        return params_values_dict
    else:
        raise ValueError


def load_chains(root, burnin: float = 0.3, nthin: int = 1, concatenate: bool = True, regex: str | None = None) -> Any:
    if isinstance(root, str) and (root.endswith(".?.txt") or root.endswith(".*.txt")):
        from getdist import loadMCSamples

        # cobaya samples
        if not concatenate:
            raise NotImplementedError
        return Chain.from_getdist(loadMCSamples(root[: -len("_?.txt")], settings={"ignore_rows": burnin}))
    if isinstance(root, (tuple, list)):
        iter = (path for path in root)
    elif regex:
        import re

        # use regex
        root = Path(root).resolve()
        pattern = re.compile(regex)
        iter = (path for path in root.iterdir() if pattern.match(path.name))
    else:
        # use glob
        root = Path(root).resolve()
        iter = (path for path in root.parent.glob(root.name))
    chains: Any = [Chain.load(path).remove_burnin(burnin)[::nthin] for path in iter]  # type: ignore
    if not concatenate:
        if any(not chain.shape for chain in chains):
            raise FileNotFoundError(f"{root} not found")
        return chains
    chains = Chain.concatenate(chains)
    if not chains.shape:
        raise FileNotFoundError(f"{root} not found")
    return chains


class FitProduct:
    def __init__(self, chain=None, profile=None, minlike=None, resample=None) -> None:
        self.chain = chain
        self.profile = profile
        self.minlike = minlike
        self.resample = resample

        self._MAP = {}
        self._ML = {}

    def _unpack_desilike_dict(self, d):
        retval = {}
        for k, v in d.items():
            try:
                retval[k] = v.item()
            except (AttributeError, ValueError):
                continue
        return retval

    def _choice_bestfit(self, profile, **kwarg):
        return profile.bestfit.choice(**kwarg)

    @property
    def rchain(self) -> Any:
        """reliable chain"""
        return self.resample if self.resample is not None else self.chain

    @property
    def MAP(self) -> dict[str, float]:
        return self.bestfit(ignore_prior=False) | self._MAP

    @MAP.setter
    def MAP(self, value: dict[str, float]):
        self._MAP = value

    @property
    def ML(self) -> dict[str, float]:
        return self.bestfit(ignore_prior=True) | self._ML

    @ML.setter
    def ML(self, value: dict[str, float]):
        self._ML = value

    def bestfit(self, ignore_prior: bool = False, **kwargs) -> dict[str, float]:
        if ignore_prior:
            return self._unpack_desilike_dict(self._choice_bestfit(self.minlike, **kwargs) if self.minlike else {})
        else:
            return self._unpack_desilike_dict(self._choice_bestfit(self.profile, **kwargs) if self.profile else {})

    def chi2min(self, ignore_prior: bool = False) -> float:
        if ignore_prior:
            if self.minlike is None:
                raise ValueError("missing minlike")
            return self.minlike.bestfit.chi2min
        else:
            if self.profile is None:
                raise ValueError("missing profile")
            return self.profile.bestfit.chi2min

    def chi2_dof(self, ignore_prior: bool = False) -> float:
        chi2 = self.chi2min(ignore_prior=ignore_prior)
        dof = self.profile.bestfit.attrs["ndof"]  # type: ignore
        return chi2 / dof

    def chi2_dof_str(self):
        chi2 = self.chi2min()
        size = self.profile.bestfit.attrs["size"]  # type: ignore
        nvaried = self.profile.bestfit.attrs["nvaried"]  # type: ignore
        chi2_dof = chi2 / (size - nvaried)
        return Rf"$\chi^2/\mathrm{{d.o.f.}}={chi2:.1f}/({size} - {nvaried})={chi2_dof:.2f}$"

    def empty(self, item: Literal["chain", "profile", "minlike", "resample", "any"] | None = None, /) -> bool:
        if item is None:
            if (self.chain is None) and (self.profile is None) and (self.minlike is None) and (self.resample is None):
                return True
        elif item == "chain" and self.chain is None:
            return True
        elif item == "profile" and self.profile is None:
            return True
        elif item == "minlike" and self.minlike is None:
            return True
        elif item == "resample" and self.resample is None:
            return True
        elif item == "any" and ((self.chain is None) or (self.profile is None) or (self.minlike is None)):
            # do not include resample here
            return True
        return False

    def params(self, **kwargs):
        return self.rchain.params(**kwargs)  # type: ignore

    @classmethod
    def load(cls, path: str, burnin=0.3, nthin=1, chain_ext="npz"):
        from desilike.samples import Profiles

        kwargs = dict.fromkeys(["chain", "profile", "minlike", "resample"])
        _path = Path(path).resolve()
        root = _path.parent
        stem = _path.name
        for k in kwargs.keys():
            try:
                if k == "chain":
                    kwargs[k] = load_chains(root, regex=rf"chain_{stem}_\d+.{chain_ext}", burnin=burnin, nthin=nthin)
                elif k == "profile":
                    kwargs[k] = Profiles.load(root / f"profile_{stem}.npy")
                elif k == "minlike":
                    kwargs[k] = Profiles.load(root / f"minlike_{stem}.npy")
                elif k == "resample":
                    kwargs[k] = load_chains(root, regex=rf"resample_{stem}_\d+.{chain_ext}", burnin=0)
                else:
                    assert False, "should not reach here"
            except FileNotFoundError:
                pass
        return cls(**kwargs)

    def mean(self, param):
        return self.rchain.mean(param)  # type: ignore

    def std(self, param):
        return self.rchain.std(param)  # type: ignore

    def interval(self, param):
        return tuple(_.item() for _ in self.rchain.interval(param))  # type: ignore

    def midi(self, param):
        lower, upper = self.interval(param)
        return 0.5 * (lower + upper)

    def ierr(self, param):
        lower, upper = self.interval(param)
        mid = self.midi(param)
        return [[upper - mid], [mid - lower]]

    def halfi(self, param):
        """Half interval."""
        lower, upper = self.interval(param)
        return 0.5 * (upper - lower)


class FitProductCollection(UserDict[Any, FitProduct]):
    @classmethod
    def load(cls, paths, names, labels=None, missing_error: Literal["raise", "print", "ignore"] = "raise", **kwargs):
        if labels is None:
            labels = names.copy()
        if len({len(_) for _ in (paths, names, labels)}) > 1:
            raise ValueError("should provide paths, names and labels with the same length")
        d: dict[str, FitProduct] = {}
        for path, name, label in zip(paths, names, labels):
            p = FitProduct.load(path=path, **kwargs)
            if p.empty():
                if missing_error == "raise":
                    raise FileNotFoundError(f"{path} not found")
                if missing_error == "print":
                    print(f"{path} not found")
            d[name] = p
        return cls(d)

    def sort(self):
        sorted_keys = set(self.keys())
        self.data = {k: self.data[k] for k in sorted_keys}
