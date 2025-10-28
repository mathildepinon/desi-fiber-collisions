import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections.abc import Container, Mapping
from typing import Any, Literal

from desi import DESIColors as DESIColors
from desi import DESIEdgeColors as DESIEdgeColors
from samples import FitProductCollection

# colors
GREY = matplotlib.cm.tab10.colors[-3]  # type: ignore
BLUE = matplotlib.cm.tab10.colors[0]  # type: ignore
ORANGE = matplotlib.cm.tab10.colors[1]  # type: ignore
GREEN = matplotlib.cm.tab10.colors[2]  # type: ignore
RED = matplotlib.cm.tab10.colors[3]  # type: ignore


def compare_power_spectrum(
    k,
    pk_ref,
    pk,
    xlim=(1e-3, 0.3),
    ylim=(1e-6, 1e-2),
    xscale="log",
    yscale="log",
    xlabel=R"$k\,[h\,\mathrm{Mpc}^{-1}]$",
    ylabel=R"$P_\ell(k)\,[h^{-1}\mathrm{Mpc}]^3$",
    fig=None,
):
    if fig is None:
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), height_ratios=[3, 1], gridspec_kw={"hspace": 0}, sharex=True)
    else:
        axes = fig.axes
    ax = axes[0]
    ax.set_xlim(*xlim)
    ax.set_xscale(xscale)
    ax.axhline(0, lw=1, c="c")
    ax.set(ylabel=ylabel)
    ax = axes[1]
    ax.set(xlabel=xlabel, ylabel=R"$|\Delta P_\ell(k)|/|P_\ell(k)|$", ylim=ylim)
    ax.set_yscale(yscale)

    ax = axes[0]
    for i, c in zip(range(len(pk_ref)), "kbg"):
        ax.plot(k, pk_ref[i], c=c)
        ax.plot(k, pk[i], c=c, ls="--")

    ax = axes[1]
    for i, c in zip(range(len(pk_ref)), "kbg"):
        y = (pk[i] - pk_ref[i]) / np.abs(pk_ref[i])
        paint_abs(k, y, ax=ax, c=c)
    return fig, axes


def group_ruler(*groups, base_scale=0.7):
    shifts = [np.linspace(-1, 1, 2 * nvalues + 1)[1:-1:2] for nvalues in groups]
    scales = [1 / nvalues for nvalues in (1,) + groups]
    scales = list(itertools.accumulate(scales, lambda a, b: a * b))
    shifts = [shift * scale * base_scale**i for i, (shift, scale) in enumerate(zip(shifts, scales))]
    retval = [shifts[-1]]
    for shift in shifts[::-1][1:]:
        new_retval = []
        for e in shift:
            for arr in retval:
                new_retval.append(e + arr)
        retval = new_retval
    return np.hstack(retval)


def paint_abs(x, y, ax=None, **kwargs):
    """Paint the 1D curve using the absolute value of ``y``, where negative values with dashed lines"""
    if ax is None:
        ax = plt.gca()
    if "c" not in kwargs and "color" not in kwargs:
        kwargs["color"] = "C0"
    not_negative = y >= 0
    nodes = np.hstack([[0], np.where(np.diff(not_negative))[0] + 1])
    if len(nodes) == 1:
        if not_negative[0]:
            ax.plot(x, y, ls="-", **kwargs)
        else:
            ax.plot(x, -y, ls="--", **kwargs)
        return ax
    it = itertools.cycle(["-", "--"] if not_negative[0] else ["--", "-"])
    for istart, istop, ls in zip(nodes[:-1], nodes[1:], it):
        if istart != 0:
            ax.plot(x[istart - 1 : istart + 1], np.abs(y[istart - 1 : istart + 1]), ls=":", **kwargs)
        ax.plot(x[istart:istop], np.abs(y[istart:istop]), ls=ls, **kwargs)
    return ax


def paint_text(ax, text, loc="lower right", x=None, y=None, va=None, ha=None, fontsize="medium", **kwargs):
    match loc:
        case "lower right":
            x = x or 0.95
            y = y or 0.05
            va = va or "bottom"
            ha = ha or "right"
        case "top right":
            x = x or 0.95
            y = y or 0.95
            va = va or "top"
            ha = ha or "right"
        case "lower left":
            x = x or 0.05
            y = y or 0.05
            va = va or "bottom"
            ha = ha or "left"
        case "top left":
            x = x or 0.05
            y = y or 0.95
            va = va or "top"
            ha = ha or "left"
        case _:
            raise NotImplementedError
    ax.text(x, y, text, transform=ax.transAxes, va=va, ha=ha, fontsize=fontsize, **kwargs)
    return ax


def paint_bestfit(g, bestfit: Mapping[str, float], color, marker="x"):
    import itertools

    for (xname, xvalue), (yname, yvalue) in itertools.permutations(bestfit.items(), 2):
        if ax := g.get_axes_for_params(xname, yname):
            ax.scatter(xvalue, yvalue, color=color, marker=marker)


def paint_markers(
    g,
    markers: Mapping[str, float],
    marker_args: Mapping[str, Any] = {"c": "k", "ls": "--", "lw": 1},
    all_params: list[str] | None = None,
):
    import itertools

    for name, value in markers.items():
        if ax := g.get_axes_for_params(name):
            ax.axvline(value, **marker_args)
    if all_params:
        all_params = all_params + list(markers.keys())
        all_params = list(set(all_params))
        for xname, xvalue in markers.items():
            for yname in all_params:
                if ax := g.get_axes_for_params(xname, yname):
                    ax.axvline(xvalue, **marker_args)
        for yname, yvalue in markers.items():
            for xname in all_params:
                if ax := g.get_axes_for_params(xname, yname):
                    ax.axhline(yvalue, **marker_args)
        return
    for (xname, xvalue), (yname, yvalue) in itertools.permutations(markers.items(), 2):
        if ax := g.get_axes_for_params(xname, yname):
            ax.axvline(xvalue, **marker_args)
            ax.axhline(yvalue, **marker_args)


def shapefit_polycorr_null_test(
    fpc_SF: FitProductCollection,
    fpc_SF_poly: FitProductCollection,
    bin_name: str,
    marker_type: Literal["MAP", "mean"] = "MAP",
    lw: float = 0.8,
    ms: float = 4.5,
    include_s0: bool = False,
    chi2_dof: float = 1.0,
    chi2_ignore_prior: bool = False,
    fig=None,
):
    color = DESIColors()[bin_name]
    ecolor = DESIEdgeColors()[bin_name]
    fig, axes = plt.subplots(
        9 + bool(include_s0), 1, figsize=(12, 8), sharex=True, gridspec_kw={"hspace": 0}, layout="constrained"
    )
    for ax in axes:
        ax.tick_params(axis="both", direction="in", right=True, top=True)
        ax.set_xticks(range(-1, 25), ["mean"] + [*range(25)])
    axes[0].axhline(1.0, lw=lw, c="k", ls="--")
    if marker_type == "MAP":
        axes[0].set_ylabel(R"$\alpha_\mathrm{iso}^\mathrm{MAP}$")
    else:
        axes[0].set_ylabel(R"$\alpha_\mathrm{iso}$")
    axes[1].set_ylabel(R"$\sigma_{\alpha_\mathrm{iso}}/\alpha_\mathrm{iso}$")

    axes[2].axhline(1.0, lw=lw, c="k", ls="--")
    if marker_type == "MAP":
        axes[2].set_ylabel(R"$\alpha_\mathrm{ap}^\mathrm{MAP}$")
    else:
        axes[2].set_ylabel(R"$\alpha_\mathrm{ap}$")
    axes[3].set_ylabel(R"$\sigma_{\alpha_\mathrm{ap}}/\alpha_\mathrm{ap}$")

    axes[4].axhline(1.0, lw=lw, c="k", ls="--")
    if marker_type == "MAP":
        axes[4].set_ylabel(R"$df^\mathrm{MAP}$")
    else:
        axes[4].set_ylabel(R"$df$")
    axes[5].set_ylabel(R"$\sigma_{df}/df$")

    axes[6].axhline(0.0, lw=lw, c="k", ls="--")
    if marker_type == "MAP":
        axes[6].set_ylabel(R"$dm^\mathrm{MAP}$")
    else:
        axes[6].set_ylabel(R"$dm$")
    axes[7].set_ylabel(R"$\sigma_{dm}/(1+dm)$")
    if include_s0:
        axes[-2].axhline(0.0, lw=lw, c="k", ls="--")
        axes[-2].set_ylabel(R"$s_0$")
    axes[-1].axhline(1.0, lw=lw, c="k", ls="--")
    axes[-1].set_ylabel(R"$\chi^2/\mathrm{d.o.f.}$")

    common = dict(c=ecolor, marker="o", ls="none", mec=ecolor, capsize=2, ms=ms, elinewidth=lw)
    axes[0].errorbar([], [], [], mfc="w", label="w/o poly corr.", **common)
    axes[0].errorbar([], [], [], mfc=color, label="w/ poly corr.", **common)
    axes[0].legend(frameon=False, ncol=2)
    axes[-1].set_xlabel("mock realisations")

    # data
    for _j, _param in zip([*range(8)][0::2], ["qiso", "qap", "df", "dm"]):
        axes[_j].fill_between([-0.5, 24.5], *fpc_SF["mean"].interval(_param), color=color, alpha=0.2, ec="none")
    if include_s0:
        axes[-2].fill_between([-0.5, 24.5], *fpc_SF_poly["mean"].interval("syst_0"), color=color, alpha=0.2, ec="none")
    _std = np.sqrt(2 / fpc_SF["mean"].chain.attrs["ndof"])  # type: ignore
    axes[-1].fill_between([-0.5, 24.5], chi2_dof - _std, chi2_dof + _std, color=color, alpha=0.2, ec="none")
    for _fpc, _offset, _mfc in [
        [fpc_SF, -0.13, "w"],
        [fpc_SF_poly, 0.13, color],
    ]:
        for _i, _name in zip(
            range(-1, 25),
            ["mean"] + [*range(25)],
        ):
            plot_common = dict(marker="o", ms=ms, mec=ecolor, mfc=_mfc, markeredgewidth=lw)
            for _j, _param in zip(
                [*range(8)][0::2] + ([-2] if (include_s0 and _fpc is fpc_SF_poly) else []),
                ["qiso", "qap", "df", "dm"] + (["syst_0"] if (include_s0 and _fpc is fpc_SF_poly) else []),
            ):
                _lower, _upper = _fpc[_name].interval(_param)
                _mid = 0.5 * (_lower + _upper)
                _yerr = [[_upper - _mid], [_mid - _lower]]
                axes[_j].errorbar(_i + _offset, _mid, _yerr, marker="None", c=ecolor, capsize=1, elinewidth=lw)
                _y = _fpc[_name].MAP[_param] if marker_type == "MAP" else _fpc[_name].mean(_param)
                axes[_j].plot(_i + _offset, _y, **plot_common)
            for _j, _param in zip([*range(8)][1::2], ["qiso", "qap", "df", "dm"]):
                _y = _fpc[_name].MAP[_param] if marker_type == "MAP" else _fpc[_name].mean(_param)
                axes[_j].plot(
                    _i + _offset, _fpc[_name].halfi(_param) / (_y + (1 if _param == "dm" else 0)), **plot_common
                )
            if _i != -1:
                axes[-1].plot(_i + _offset, _fpc[_name].chi2_dof(ignore_prior=chi2_ignore_prior), **plot_common)
    return fig, axes


def shapefit_polycorr_null_test_scatter(
    fpc_ref: FitProductCollection, fpc: FitProductCollection, bin_name: str, ireal_exclude: Container[int] = []
):
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), layout="constrained")
    axes[0, 0].set_xlabel(R"$\alpha_\mathrm{ap}^\mathrm{MAP}\mathrm{\ poly\ corr.} - \alpha_\mathrm{ap}^\mathrm{MAP}$")
    axes[0, 0].set_ylabel(R"$\alpha_\mathrm{iso}^\mathrm{MAP}\mathrm{\ poly\ corr.} -\alpha_\mathrm{iso}^\mathrm{MAP}$")
    axes[0, 0].axhline(0, ls="--", c="k", lw=1)
    axes[0, 0].axvline(0, ls="--", c="k", lw=1)

    axes[0, 1].set_xlabel(R"$\sigma_{\alpha_\mathrm{ap}}\mathrm{\ poly\ corr.} / \sigma_{\alpha_\mathrm{ap}} - 1$")
    axes[0, 1].set_ylabel(R"$\sigma_{\alpha_\mathrm{iso}}\mathrm{\ poly\ corr.} / \sigma_{\alpha_\mathrm{iso}} - 1$")
    axes[0, 1].axhline(0, ls="--", c="k", lw=1)
    axes[0, 1].axvline(0, ls="--", c="k", lw=1)

    axes[1, 0].set_xlabel(R"$dm^\mathrm{MAP}\mathrm{\ poly\ corr.} -dm^\mathrm{MAP}$")
    axes[1, 0].set_ylabel(R"$df^\mathrm{MAP}\mathrm{\ poly\ corr.} -df^\mathrm{MAP}$")
    axes[1, 0].axhline(0, ls="--", c="k", lw=1)
    axes[1, 0].axvline(0, ls="--", c="k", lw=1)

    axes[1, 1].set_xlabel(R"$\sigma_{dm}\mathrm{\ poly\ corr.} / \sigma_{dm} - 1$")
    axes[1, 1].set_ylabel(R"$\sigma_{df}\mathrm{\ poly\ corr.} / \sigma_{df} - 1$")
    axes[1, 1].axhline(0, ls="--", c="k", lw=1)
    axes[1, 1].axvline(0, ls="--", c="k", lw=1)
    paint_text(axes[0, 0], bin_name, "top right", fontsize="large")
    color = DESIColors()[bin_name]
    ecolor = DESIEdgeColors()[bin_name]

    for ireal in range(25):
        if ireal in ireal_exclude:
            continue
        axes[0, 0].plot(
            fpc[ireal].MAP["qap"] - fpc_ref[ireal].MAP["qap"],
            fpc[ireal].MAP["qiso"] - fpc_ref[ireal].MAP["qiso"],
            marker="o",
            mfc=color,
            mec=ecolor,
        )
        axes[0, 1].plot(
            fpc[ireal].halfi("qap") / fpc_ref[ireal].halfi("qap") - 1,
            fpc[ireal].halfi("qiso") / fpc_ref[ireal].halfi("qiso") - 1,
            marker="o",
            mfc=color,
            mec=ecolor,
        )
        axes[1, 0].plot(
            fpc[ireal].MAP["dm"] - fpc_ref[ireal].MAP["dm"],
            fpc[ireal].MAP["df"] - fpc_ref[ireal].MAP["df"],
            marker="o",
            mfc=color,
            mec=ecolor,
        )
        axes[1, 1].plot(
            fpc[ireal].halfi("dm") / fpc_ref[ireal].halfi("dm") - 1,
            fpc[ireal].halfi("df") / fpc_ref[ireal].halfi("df") - 1,
            marker="o",
            mfc=color,
            mec=ecolor,
        )

    axes[0, 0].errorbar(
        [fpc["mean"].MAP["qap"] - fpc_ref["mean"].MAP["qap"]],
        [fpc["mean"].MAP["qiso"] - fpc_ref["mean"].MAP["qiso"]],
        xerr=fpc["mean"].ierr("qap"),
        yerr=fpc["mean"].ierr("qiso"),
        marker="o",
        color=ecolor,
        mfc="w",
        capsize=5,
    )
    axes[0, 1].errorbar(
        [fpc["mean"].halfi("qap") / fpc_ref["mean"].halfi("qap") - 1],
        [fpc["mean"].halfi("qiso") / fpc_ref["mean"].halfi("qiso") - 1],
        marker="o",
        color=ecolor,
        mfc="w",
        capsize=5,
    )
    axes[1, 0].errorbar(
        [fpc["mean"].MAP["dm"] - fpc_ref["mean"].MAP["dm"]],
        [fpc["mean"].MAP["df"] - fpc_ref["mean"].MAP["df"]],
        xerr=fpc["mean"].ierr("dm"),
        yerr=fpc["mean"].ierr("df"),
        marker="o",
        color=ecolor,
        mfc="w",
        capsize=5,
    )
    axes[1, 1].errorbar(
        [fpc["mean"].halfi("dm") / fpc_ref["mean"].halfi("dm") - 1],
        [fpc["mean"].halfi("df") / fpc_ref["mean"].halfi("df") - 1],
        marker="o",
        color=ecolor,
        mfc="w",
        capsize=5,
    )
    return fig, axes


def shapefit_compare_bins(fpc_ref: FitProductCollection, fpc: FitProductCollection):
    fig, axes = plt.subplots(2, 4, figsize=(12, 4), gridspec_kw={"hspace": 0}, sharex=True, layout="constrained")
    for ax in axes.flat:
        ax.tick_params(axis="both", direction="in", right=True, top=True)
        ax.set_xlim(-0.5, len(fpc_ref) - 0.5)
    for ax in axes[-1].flat:
        ax.set_xticks(range(len(fpc_ref)), fpc_ref.keys())
        ax.set_xticklabels(fpc_ref.keys(), rotation=30, ha="right", va="top", rotation_mode="anchor", fontsize=14)
        ax.axhline(0.0, lw=1, c="k", ls="--")
        ax.fill_between([-0.5, len(fpc_ref) - 0.5], -0.2, 0.2, color="c", alpha=0.2, ec="none")
        ax.axhline(-0.4, c="r", ls="--", lw=1)
        ax.axhline(0.4, c="r", ls="--", lw=1)
    for ax in axes[1, :].flat:
        ax.set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
        ax.set_yticklabels([-0.4, -0.2, 0, 0.2, 0.4])
    axes[0, 0].set_ylabel(R"$\alpha_\mathrm{iso}^\mathrm{MAP}$", fontsize=14)
    axes[0, 0].axhline(1.0, lw=1, c="k", ls="--")
    axes[0, 1].set_ylabel(R"$\alpha_\mathrm{AP}^\mathrm{MAP}$", fontsize=14)
    axes[0, 1].axhline(1.0, lw=1, c="k", ls="--")
    axes[0, 2].set_ylabel(R"$(f/f_{\rm fid})^\mathrm{MAP}$", fontsize=14)
    axes[0, 2].axhline(1.0, lw=1, c="k", ls="--")
    axes[0, 3].set_ylabel(R"$m^\mathrm{MAP}$", fontsize=14)
    axes[0, 3].axhline(0.0, lw=1, c="k", ls="--")
    axes[1, 0].set_ylabel(R"$\Delta\alpha_\mathrm{iso}^\mathrm{MAP}/\sigma_{\alpha_\mathrm{iso}}$", fontsize=14)
    axes[1, 1].set_ylabel(R"$\Delta\alpha_\mathrm{AP}^\mathrm{MAP}/\sigma_{\alpha_\mathrm{AP}}$", fontsize=14)
    axes[1, 2].set_ylabel(R"$\Delta (f/f_{\rm fid})^\mathrm{MAP}/\sigma_{(f/f_{\rm fid})}$", fontsize=14)
    axes[1, 3].set_ylabel(R"$\Delta m^\mathrm{MAP}/\sigma_{m}$", fontsize=14)

    for _fpc, _offset in ((fpc_ref, -0.13), (fpc, 0.13)):
        for _ax, _param in zip(axes[0], ["qiso", "qap", "df", "dm"]):
            for _j, (_name, _p) in enumerate(_fpc.items()):
                ecolor = DESIEdgeColors()[_name]
                color = DESIColors()[_name]
                _mfc = "w" if _fpc is fpc_ref else color

                _ax.errorbar(_j + _offset, _p.midi(_param), _p.ierr(_param), marker="None", c=ecolor, capsize=2)
                _ax.plot(_j + _offset, _p.MAP[_param], marker="o", ms=5, mec=ecolor, mfc=_mfc)
    for _ax, _param in zip(axes[1], ["qiso", "qap", "df", "dm"]):
        for _j, _name in enumerate(_fpc.keys()):
            ecolor = DESIEdgeColors()[_name]
            color = DESIColors()[_name]
            _ax.plot(
                _j,
                (fpc[_name].MAP[_param] - fpc_ref[_name].MAP[_param]) / fpc_ref[_name].halfi(_param),
                marker="o",
                ms=5,
                mec=ecolor,
                mfc=color,
            )
    return fig, axes


def fullmodel_compare_bins(fpc_ref: FitProductCollection, fpc: FitProductCollection):
    from cosmoprimo.fiducial import DESI

    _cosmo = DESI()
    TRUTH = dict(qpar=1, qper=1, qiso=1, qap=1, dm=0, dn=0, df=1) | {
        "Omega_m": _cosmo.Omega0_m,
        "logA": np.log(10**10 * _cosmo.A_s),
        "sigma8_m": _cosmo.sigma8_m,
        "h": _cosmo.h,
        "H0": 100 * _cosmo.h,
        "omega_cdm": _cosmo["omega_cdm"],
        "omega_b": _cosmo["omega_b"],
        "n_s": _cosmo["n_s"],
        "w0_fld": -1.0,
        "wa_fld": 0.0,
        "sigmabao": 0.0,
        "syst_0": 0.0,
        "syst_1": 1.0,
    }

    fig, axes = plt.subplots(2, 4, figsize=(12, 4), gridspec_kw={"hspace": 0}, sharex=True, layout="constrained")
    for ax in axes.flat:
        ax.tick_params(axis="both", direction="in", right=True, top=True)
        ax.set_xlim(-0.5, len(fpc_ref) - 0.5)
    for ax in axes[-1].flat:
        ax.set_xticks(range(len(fpc_ref)), fpc_ref.keys())
        ax.set_xticklabels(fpc_ref.keys(), rotation=30, ha="right", va="top", rotation_mode="anchor", fontsize=14)
        ax.axhline(0.0, lw=1, c="k", ls="--")
        ax.fill_between([-0.5, len(fpc_ref) - 0.5], -0.2, 0.2, color="c", alpha=0.2, ec="none")
        ax.axhline(-0.4, c="r", ls="--", lw=1)
        ax.axhline(0.4, c="r", ls="--", lw=1)
    axes[0, 0].set_ylabel(R"$h^\mathrm{MAP}$", fontsize=14)
    axes[0, 0].axhline(TRUTH["h"], lw=1, c="k", ls="--")
    axes[0, 1].set_ylabel(R"$\omega_{cdm}^\mathrm{MAP}$", fontsize=14)
    axes[0, 1].axhline(TRUTH["omega_cdm"], lw=1, c="k", ls="--")
    axes[0, 2].set_ylabel(R"$\sigma_8^\mathrm{MAP}$", fontsize=14)
    axes[0, 2].axhline(TRUTH["sigma8_m"], lw=1, c="k", ls="--")
    axes[0, 3].set_ylabel(R"$\log A^\mathrm{MAP}$", fontsize=14)
    axes[0, 3].axhline(TRUTH["logA"], lw=1, c="k", ls="--")
    axes[1, 0].set_ylabel(R"$\Delta h^\mathrm{MAP}/\sigma_{h}$", fontsize=14)
    axes[1, 1].set_ylabel(R"$\Delta\omega_{cdm}^\mathrm{MAP}/\sigma_{\omega_{cdm}}$", fontsize=14)
    axes[1, 2].set_ylabel(R"$\Delta \sigma_8^\mathrm{MAP}/\sigma_{\sigma_8}$", fontsize=14)
    axes[1, 3].set_ylabel(R"$\Delta \log A^\mathrm{MAP}/\sigma_{\log A}$", fontsize=14)

    for _fpc, _offset in ((fpc_ref, -0.13), (fpc, 0.13)):
        for _ax, _param in zip(axes[0], ["h", "omega_cdm", "sigma8_m", "logA"]):
            for _j, (_name, _p) in enumerate(_fpc.items()):
                ecolor = DESIEdgeColors()[_name]
                color = DESIColors()[_name]
                _mfc = "w" if _fpc is fpc_ref else color

                _ax.errorbar(_j + _offset, _p.midi(_param), _p.ierr(_param), marker="None", c=ecolor, capsize=2)
                _ax.plot(_j + _offset, _p.MAP[_param], marker="o", ms=5, mec=ecolor, mfc=_mfc)
    for _ax, _param in zip(axes[1], ["h", "omega_cdm", "sigma8_m", "logA"]):
        for _j, _name in enumerate(_fpc.keys()):
            ecolor = DESIEdgeColors()[_name]
            color = DESIColors()[_name]
            _ax.plot(
                _j,
                (fpc[_name].MAP[_param] - fpc_ref[_name].MAP[_param]) / fpc_ref[_name].halfi(_param),
                marker="o",
                ms=5,
                mec=ecolor,
                mfc=color,
            )
    return fig, axes


def fullmodel_polycorr_null_test_scatter(
    fpc_ref: FitProductCollection, fpc: FitProductCollection, bin_name: str, ireal_exclude: Container[int] = []
):
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), layout="constrained")
    axes[0, 0].set_xlabel(R"$h^\mathrm{MAP}\mathrm{\ poly\ corr.} - h^\mathrm{MAP}$")
    axes[0, 0].set_ylabel(R"$\Omega_m^\mathrm{MAP}\mathrm{\ poly\ corr.} - \Omega_m^\mathrm{MAP}$")
    axes[0, 0].axhline(0, ls="--", c="k", lw=1)
    axes[0, 0].axvline(0, ls="--", c="k", lw=1)

    axes[0, 1].set_xlabel(R"$\sigma_{h}\mathrm{\ poly\ corr.} / \sigma_{h} - 1$")
    axes[0, 1].set_ylabel(R"$\sigma_{\Omega_m}\mathrm{\ poly\ corr.} / \sigma_{\Omega_m} - 1$")
    axes[0, 1].axhline(0, ls="--", c="k", lw=1)
    axes[0, 1].axvline(0, ls="--", c="k", lw=1)

    axes[1, 0].set_xlabel(R"$\sigma_8^\mathrm{MAP}\mathrm{\ poly\ corr.} -\sigma_8^\mathrm{MAP}$")
    axes[1, 0].set_ylabel(R"$\omega_{cdm}^\mathrm{MAP}\mathrm{\ poly\ corr.} -\omega_{cdm}^\mathrm{MAP}$")
    axes[1, 0].axhline(0, ls="--", c="k", lw=1)
    axes[1, 0].axvline(0, ls="--", c="k", lw=1)

    axes[1, 1].set_xlabel(R"$\sigma_{\sigma_8}\mathrm{\ poly\ corr.} / \sigma_{\sigma_8} - 1$")
    axes[1, 1].set_ylabel(R"$\sigma_{\omega_{cdm}}\mathrm{\ poly\ corr.} / \sigma_{\omega_{cdm}} - 1$")
    axes[1, 1].axhline(0, ls="--", c="k", lw=1)
    axes[1, 1].axvline(0, ls="--", c="k", lw=1)
    paint_text(axes[0, 0], bin_name, "top right", fontsize="large")
    color = DESIColors()[bin_name]
    ecolor = DESIEdgeColors()[bin_name]

    for ireal in range(25):
        if ireal in ireal_exclude:
            continue
        try:
            axes[0, 0].plot(
                fpc[ireal].MAP["h"] - fpc_ref[ireal].MAP["h"],
                fpc[ireal].MAP["Omega_m"] - fpc_ref[ireal].MAP["Omega_m"],
                marker="o",
                mfc=color,
                mec=ecolor,
            )
            axes[0, 1].plot(
                fpc[ireal].halfi("h") / fpc_ref[ireal].halfi("h") - 1,
                fpc[ireal].halfi("Omega_m") / fpc_ref[ireal].halfi("Omega_m") - 1,
                marker="o",
                mfc=color,
                mec=ecolor,
            )
            axes[1, 0].plot(
                fpc[ireal].MAP["sigma8_m"] - fpc_ref[ireal].MAP["sigma8_m"],
                fpc[ireal].MAP["omega_cdm"] - fpc_ref[ireal].MAP["omega_cdm"],
                marker="o",
                mfc=color,
                mec=ecolor,
            )
            axes[1, 1].plot(
                fpc[ireal].halfi("sigma8_m") / fpc_ref[ireal].halfi("sigma8_m") - 1,
                fpc[ireal].halfi("omega_cdm") / fpc_ref[ireal].halfi("omega_cdm") - 1,
                marker="o",
                mfc=color,
                mec=ecolor,
            )
        except Exception:
            pass

    axes[0, 0].errorbar(
        [fpc["mean"].MAP["h"] - fpc_ref["mean"].MAP["h"]],
        [fpc["mean"].MAP["Omega_m"] - fpc_ref["mean"].MAP["Omega_m"]],
        xerr=fpc["mean"].ierr("h"),
        yerr=fpc["mean"].ierr("Omega_m"),
        marker="o",
        color=ecolor,
        mfc="w",
        capsize=5,
    )
    axes[0, 1].errorbar(
        [fpc["mean"].halfi("h") / fpc_ref["mean"].halfi("h") - 1],
        [fpc["mean"].halfi("Omega_m") / fpc_ref["mean"].halfi("Omega_m") - 1],
        marker="o",
        color=ecolor,
        mfc="w",
        capsize=5,
    )
    axes[1, 0].errorbar(
        [fpc["mean"].MAP["sigma8_m"] - fpc_ref["mean"].MAP["sigma8_m"]],
        [fpc["mean"].MAP["omega_cdm"] - fpc_ref["mean"].MAP["omega_cdm"]],
        xerr=fpc["mean"].ierr("sigma8_m"),
        yerr=fpc["mean"].ierr("omega_cdm"),
        marker="o",
        color=ecolor,
        mfc="w",
        capsize=5,
    )
    axes[1, 1].errorbar(
        [fpc["mean"].halfi("sigma8_m") / fpc_ref["mean"].halfi("sigma8_m") - 1],
        [fpc["mean"].halfi("omega_cdm") / fpc_ref["mean"].halfi("omega_cdm") - 1],
        marker="o",
        color=ecolor,
        mfc="w",
        capsize=5,
    )
    return fig, axes
