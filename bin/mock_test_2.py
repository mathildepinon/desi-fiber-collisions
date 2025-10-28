# code by Ruiyang

from __future__ import annotations
import argparse
import logging
import numpy as np
from contextlib import contextmanager
from desilike import setup_logging
from desilike.emulators import Emulator, EmulatedCalculator, TaylorEmulatorEngine
from desilike.likelihoods import ObservablesGaussianLikelihood
from desilike.observables import ObservableCovariance
from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
from desilike.profilers import MinuitProfiler
from desilike.samplers import NUTSSampler
from desilike.samples import Profiles
from desilike.theories.galaxy_clustering import (
    ShapeFitPowerSpectrumTemplate,
    DirectPowerSpectrumTemplate,
    LPTVelocileptorsTracerPowerSpectrumMultipoles,
    REPTVelocileptorsTracerPowerSpectrumMultipoles,
    FOLPSAXTracerPowerSpectrumMultipoles,
)
from desilike.theories.primordial_cosmology import Cosmoprimo
from mpi4py import MPI
from pathlib import Path
from typing import Any, cast, Final, Literal, NamedTuple

BBN_CENTER = 0.02237
BBN_PRIOR = 0.00055
COMM = MPI.COMM_WORLD
MISSING: Final = object()


def assert_never():
    assert False, "should never reach here!"


def get_mpi_logger(name: str, mpicomm=MPI.COMM_WORLD) -> MPILogger:
    """Return a logger with the specified name or, if name is None, return a logger which is the root logger of the hierarchy."""
    raw_logger_class = logging.getLoggerClass()
    logging.setLoggerClass(MPILogger)
    logger = cast(MPILogger, MPILogger.manager.getLogger(name))
    logger.mpicomm = mpicomm
    logging.setLoggerClass(raw_logger_class)
    return logger


class MPILogger(logging.Logger):
    """Custom logger class that logs messages based on the rank of the MPI process.

    Parameters
    ----------
    name : str
        Name of logger.
    level : int, optional
        Logging level, default `logging.NOTSET`.
    mpicomm : `mpi4py.MPI.Intracomm`, optional
        Communicator, default `mpi4py.MPI.COMM_WORLD`.
    """

    def __init__(self, name: str, level=logging.NOTSET, mpicomm=MPI.COMM_WORLD):
        super().__init__(name, level=level)
        self.mpicomm = mpicomm

    def debug(self, *args, rank: int | None = 0, **kwargs):
        if rank is None or rank == self.mpicomm.rank:
            super().debug(*args, **kwargs)

    def info(self, *args, rank: int | None = 0, **kwargs):
        if rank is None or rank == self.mpicomm.rank:
            super().info(*args, **kwargs)

    def warning(self, *args, rank: int | None = 0, **kwargs):
        if rank is None or rank == self.mpicomm.rank:
            super().warning(*args, **kwargs)

    def warn(self, *args, rank: int | None = 0, **kwargs):
        if rank is None or rank == self.mpicomm.rank:
            super().warn(*args, **kwargs)

    def exception(self, *args, rank: int | None = 0, **kwargs):
        if rank is None or rank == self.mpicomm.rank:
            super().exception(*args, **kwargs)

    def critical(self, *args, rank: int | None = 0, **kwargs):
        if rank is None or rank == self.mpicomm.rank:
            super().critical(*args, **kwargs)

    def fatal(self, *args, rank: int | None = 0, **kwargs):
        if rank is None or rank == self.mpicomm.rank:
            super().fatal(*args, **kwargs)

    def log(self, level, msg, *args, rank: int | None = 0, **kwargs):
        if rank is None or rank == self.mpicomm.rank:
            super().log(level=level, msg=msg, *args, **kwargs)


LOGGER = get_mpi_logger("mock test")


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
    from desilike.samples import Chain

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


def print_priors(calculator, mpicomm=None):
    if mpicomm is None:
        from mpi4py import MPI

        mpicomm = MPI.COMM_WORLD
    if mpicomm.rank == 0:
        print(f"{'param':15} {'prior':50} {'reference':50} derived")
        for p in calculator.all_params.select(varied=True):
            print(f"{p.name:15} {str(p.prior):50} {str(p.ref):50} {p.derived}")


def expand_bin_name(bin_name: str, is_data: bool = False):
    bin_name = bin_name.upper()
    if bin_name == "BGS":
        return "BGS_BRIGHT-21.5" if is_data else "BGS", "0.1_0.4"
    if bin_name == "LRG1":
        return "LRG", "0.4_0.6"
    if bin_name == "LRG2":
        return "LRG", "0.6_0.8"
    if bin_name == "LRG3":
        return "LRG", "0.8_1.1"
    if bin_name == "ELG1":
        return "ELG_LOPnotqso" if is_data else "ELG", "0.8_1.1"
    if bin_name == "ELG2":
        return "ELG_LOPnotqso" if is_data else "ELG", "1.1_1.6"
    if bin_name == "QSO":
        return "QSO", "0.8_2.1"
    raise ValueError


def get_zeff(bin_name):
    bin_name = bin_name.upper()
    match bin_name:
        case "BGS":
            return 0.2
        case "LRG1":
            return 0.5
        case "LRG2":
            return 0.8
        case "LRG3":
            return 0.8
        case "ELG1":
            return 0.95
        case "ELG2":
            return 1.325
        case "QSO":
            return 1.4
    raise ValueError


def get_template(template, zeff, prior=None, bbn_prior=BBN_PRIOR, bbn_center=BBN_CENTER):
    if template == "SF":
        template = ShapeFitPowerSpectrumTemplate(z=zeff, apmode="qisoqap", fiducial="DESI")
        # default prior for qiso and qap
        template.params["qiso"].update(prior={"limits": [0.8, 1.2]})
        template.params["qap"].update(prior={"limits": [0.8, 1.2]})
        # `delta` suggested by Arnaud
        template.init.params["qiso"].update(delta=0.02)
        template.init.params["qap"].update(delta=0.02)
        template.init.params["df"].update(delta=0.05)
        # limit the range of dm (otherwise it's easy to have bimodial posterior)
        template.init.params["dm"].update(prior={"limits": [-0.8, 0.8]})
        if prior == "freedn":
            template.init.params["dn"].update(prior=None, fixed=False)
    elif template == "FM":
        cosmo = Cosmoprimo(fiducial="DESI")
        cosmo.init.params["omega_b"].update(prior={"dist": "norm", "loc": bbn_center, "scale": bbn_prior})
        cosmo.init.params["tau_reio"].update(fixed=True)
        # backward compatibility (consistent with DirectPowerSpectrumTemplate)
        cosmo.init.params["h"].update(prior={"dist": "uniform", "limits": [0.1, 1.0]})
        cosmo.init.params["omega_cdm"].update(delta=0.01)
        cosmo.init.params["n_s"].update(fixed=True)
        if prior == "ns004":
            cosmo.init.params["n_s"].update(fixed=False, prior={"dist": "norm", "loc": 0.9649, "scale": 0.042})
        # derived parameters
        cosmo.init.params["sigma8_m"] = {"derived": True, "fixed": False, "latex": R"\sigma_8"}
        cosmo.init.params["Omega_m"] = {"derived": True, "fixed": False, "latex": R"\Omega_m"}
        cosmo.init.params["rs_drag"] = {"derived": True, "fixed": False, "latex": R"r_s"}
        template = DirectPowerSpectrumTemplate(z=zeff, cosmo=cosmo)
    else:
        assert False, "should not reach"
    return template


def get_theory(bin_name, template, model: Literal["LPT", "EPT", "folpsax"] = "LPT"):
    bin_name = bin_name.upper()
    if model == "LPT":
        theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(
            freedom="max", prior_basis="physical", tracer=expand_bin_name(bin_name)[0]
        )
    elif model == "EPT":
        theory = REPTVelocileptorsTracerPowerSpectrumMultipoles(
            freedom="max", prior_basis="physical", tracer=expand_bin_name(bin_name)[0]
        )
    elif model == "folpsax":
        theory = FOLPSAXTracerPowerSpectrumMultipoles(
            freedom="max", prior_basis="physical", tracer=expand_bin_name(bin_name)[0]
        )
    else:
        assert_never()
    LOGGER.info("Using {} model".format(model))
    theory.init.update(template=template)
    # fix alpha4p and sn4p: fit to monopole + quadrupole
    theory.params["alpha4p"].update(fixed=True, prior={"dist": "norm", "loc": 0.0, "scale": 50.0})
    theory.params["sn4p"].update(fixed=True)
    if bin_name == "ELG2":
        for param in theory.params.select(basename=["b1p"]):
            param.update(ref={"dist": "norm", "loc": 0.6, "scale": param.ref.scale})
    if bin_name == "BGS":
        for param in theory.params.select(basename=["b1p"]):
            param.update(ref={"dist": "norm", "loc": 1.2, "scale": param.ref.scale})
    return theory


def get_observable(datainfo: DataInfo, theory, logger=LOGGER):
    data_xlim = {0: [0.02, 0.20, 0.005], 2: [0.02, 0.20, 0.005]}
    covariance = ObservableCovariance.load(datainfo.covariance)
    observable = TracerPowerSpectrumMultipolesObservable(
        data=datainfo.data,
        covariance=covariance,
        klim=data_xlim,
        wmatrix=datainfo.wmatrix,
        kin=np.arange(0.001, 0.35, 0.002),
    )
    observable.init.update(theory=theory)
    return observable


def get_likelihood(observable, marg: bool = True, logger=LOGGER):
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    likelihood()
    if marg:
        for p in likelihood.all_params.select(basename=["alpha*", "c*", "sn*"]):
            if p.varied:
                p.update(derived=".auto")
    return likelihood


class DataInfo(NamedTuple):
    zeff: float
    data: Any
    covariance: Any
    wmatrix: str

    @classmethod
    def get_mock(
        cls,
        bin_name: str,
        version: str = "v4.2",
        mock_type: str = "altMTL",
        ireal: int | None = None,
        rotated: bool = True,
        corrected: bool = True,
        thetacut: bool = True,
        cov_type: str = "EZmock",
    ):
        ctracer, redshift = expand_bin_name(bin_name, is_data=True)
        if "ELG" in ctracer and mock_type != "altMTL":
            ctracer = "ELG_LOP"
        redshift = redshift.replace("_", "-")
        zeff = get_zeff(bin_name=bin_name)

        match (rotated, corrected):
            case (True, False):
                infix = "rotated_"
                infix_path = "/rotated/"
            case (False, True):
                infix = "corrected_"
                infix_path = "/corrected/"
            case (True, True):
                infix = "rotated_corrected_"
                infix_path = "/rotated_corrected/"
            case _:
                infix = ""
                infix_path = "/"
        root = Path("/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/desipipe")
        if "BGS" == bin_name:
            root /= "BGS_v1"
        else:
            root /= version.replace(".", "_")
        root /= mock_type.lower()

        filename = "pkpoles_{}{}_GCcomb_z{}".format(infix, ctracer, redshift)
        if thetacut:
            filename += "_thetacut0.05"
        filename += ".npy"
        if ireal is None:
            ireals = [*range(25)]
        else:
            ireals = [ireal]
        data = [str(root / f"baseline_2pt/mock{_}/pk/{infix_path}{filename}") for _ in ireals]

        cov_path = root / "cov_2pt"
        if cov_type == "EZmock":
            cov_path /= "ezmock/v1"
        else:
            cov_path /= "thecov/v1.2"
        if rotated:
            cov_path /= "rotated"
        cov_path /= "covariance_{}power_{}_GCcomb_z{}_default_FKP_lin_thetacut0.05.npy".format(  # consistently use thetacut here
            "rotated_" if rotated else "", ctracer, redshift
        )
        covariance = str(cov_path)

        wmatrix_path = root / "baseline_2pt/merged/pk/wmatrix_smooth_{}{}_GCcomb_z{}{}.npy".format(
            "rotated_" if rotated else "", ctracer, redshift, "_thetacut0.05" if thetacut else ""
        )
        return cls(zeff=zeff, data=data, covariance=covariance, wmatrix=str(wmatrix_path))


if __name__ == "__main__":

    def bool_type(x):
        if x in ["True", "true", True]:
            return True
        if x in ["False", "false", False]:
            return False
        raise ValueError

    parser = argparse.ArgumentParser()
    bin_names = ["BGS", "LRG1", "LRG2", "LRG3", "ELG1", "ELG2", "QSO"]
    parser.add_argument("--bin_name", type=str.upper, choices=bin_names + [_.lower() for _ in bin_names], required=True)
    parser.add_argument("--version", choices=["v4.2"], default="v4.2", help="mock version, default: %(default)s")
    parser.add_argument("--mock-type", choices=["complete", "ffa", "altMTL"], required=True, help="mock type")
    parser.add_argument("--ireal", type=int, default=None, help="default to use all realisations")
    parser.add_argument("--rotated", type=bool_type, default=True, help="default: %(default)s")
    parser.add_argument(
        "--corrected", type=bool_type, default=True, help="whether to correct RIC or not, default: %(default)s"
    )
    parser.add_argument("--thetacut", type=bool_type, default=True, help="default: %(default)s")
    parser.add_argument("--cov-type", choices=["EZmock", "analytic"], default="EZmock", help="default: %(default)s")
    parser.add_argument("--prior", choices=["ns004", "freedn"], default=None, help="default: %(default)s")
    parser.add_argument(
        "--todo",
        nargs="+",
        choices=["emulate", "sample", "profile", "minlike", "stats", "resample"],
        default=["sample", "profile", "minlike", "stats"],
        help="default is %(default)s",
    )
    parser.add_argument("--template", choices=["FM", "SF"], default="SF", help="default is %(default)s")
    parser.add_argument("--output", type=str, required=True, help="in the form of /path/to/output_name")
    parser.add_argument(
        "--emulator-dir",
        type=Path,
        default=Path("/global/cfs/cdirs/desi/users/ruiyang/Y1/mocks/SecondGenMocks/emulators/velocileptors/LPT"),
        help="directory used to read/load emulators, default is %(default)s",
    )
    parser.add_argument("--model", choices=["EPT", "LPT", "folpsax"], default="LPT", help="model to use")
    parser.add_argument("--sampler", choices=["nuts", "cobaya"], default="nuts", help="sampler to use")
    parser.add_argument("--dryrun", action="store_true")
    args = parser.parse_args()
    setup_logging()
    LOGGER.info(f"output={args.output}")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~init~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    datainfo = DataInfo.get_mock(
        bin_name=args.bin_name,
        version=args.version,
        mock_type=args.mock_type,
        ireal=args.ireal,
        rotated=args.rotated,
        corrected=args.corrected,
        thetacut=args.thetacut,
        cov_type=args.cov_type,
    )
    template = get_template(template=args.template, zeff=datainfo.zeff, prior=args.prior)
    theory = get_theory(bin_name=args.bin_name, template=template, model=args.model)
    observable = get_observable(datainfo=datainfo, theory=theory, logger=LOGGER)
    LOGGER.info(args.version + " " + args.mock_type)
    LOGGER.info("{} {}".format(*expand_bin_name(args.bin_name)))
    LOGGER.info(f"shotnoise={observable.shotnoise}")
    likelihood = get_likelihood(observable=observable, marg=True, logger=LOGGER)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print_priors(likelihood)
    if args.dryrun:
        exit()

    if "emulate" in args.todo:
        pt = likelihood.observables[0].wmatrix.theory.pt
        LOGGER.info("emulator priors:")
        print_priors(pt)
        save_fn = args.emulator_dir / f"emulator_{args.template}_{args.bin_name}.npy"
        LOGGER.info(f"emulator will be saved to {str(save_fn)}")
        # 4-th order taylor expansion
        emulator = Emulator(pt, engine=TaylorEmulatorEngine(method="finite", order=4))  # type: ignore
        emulator.set_samples()
        emulator.fit()
        emulator.save(save_fn)
        COMM.barrier()

    with log_root_only(mpicomm=COMM):
        pt = EmulatedCalculator.load(args.emulator_dir / f"emulator_{args.template}_{args.bin_name}.npy")
    if args.prior is None:
        for param in pt.params.select(basename=["dn", "n_s"]):
            param.update(fixed=True)
    for param in pt.params.select(basename=["qiso", "qap"]):
        param.update(prior={"dist": "uniform", "limits": [0.8, 1.2]})
    for param in pt.params.select(basename=["dm"]):
        param.update(prior={"dist": "uniform", "limits": [-0.8, 0.8]})
    # use emulator
    theory.init.update(template=None, pt=pt)
    # init again because the analytical marginalisation setting lost
    for p in likelihood.all_params.select(basename=["alpha*", "c*", "sn*"]):
        if p.varied:
            p.update(derived=".auto")
    likelihood()
    LOGGER.info("Priors w/ emulator:")
    print_priors(likelihood)

    chain = None
    _path = Path(args.output)
    if "sample" in args.todo:
        save_fn = [_path.parent / ("chain_" + _path.name + f"_{i}.npz") for i in range(COMM.size)]
        sampler = NUTSSampler(likelihood, chains=COMM.size, save_fn=save_fn, ref_scale=0.05)
        check = {"max_eigen_gr": 0.01} | ({} if args.bin_name == "BGS" else {"min_ess": 1500})
        chain = sampler.run(min_iterations=2000, check=check)
        COMM.barrier()

    if "profile" in args.todo:
        save_fn = _path.parent / ("profile_" + _path.name)
        LOGGER.info("Priors for profile")
        print_priors(likelihood)
        try:
            with log_root_only(mpicomm=COMM):
                chain = load_chains(_path.parent, regex=("chain_" + _path.name + r"_\d+.npz"))
        except FileNotFoundError:
            pass
        profiler = MinuitProfiler(likelihood, covariance=chain, save_fn=save_fn, rescale=True)
        profiler.maximize(niterations=10)
        COMM.barrier()

    if "minlike" in args.todo:
        save_fn = _path.parent / ("minlike_" + _path.name)
        for param in likelihood.all_params.select(basename=["b*", "alpha*", "sn*"]):
            if param.prior.dist == "norm":
                param.update(prior=None)
        LOGGER.info("Priors for minlike")
        print_priors(likelihood)
        try:
            with log_root_only(mpicomm=COMM):
                chain = load_chains(_path.parent, regex=("chain_" + _path.name + r"_\d+.npz"))
        except FileNotFoundError:
            pass
        profiler = MinuitProfiler(likelihood, covariance=chain, save_fn=save_fn, rescale=True)
        profiler.maximize(niterations=10)
        COMM.barrier()

    if "resample" in args.todo:
        try:
            from imsys.sampler import Resampler
        except ImportError:
            pass
        else:
            with log_root_only(mpicomm=COMM):
                chains = load_chains(_path.parent, regex=("chain_" + _path.name + r"_\d+.npz"), concatenate=False)
            theory_noemu = get_theory(bin_name=args.bin_name, template=template, model=args.model)
            observable_noemu = get_observable(datainfo=datainfo, theory=theory_noemu, logger=LOGGER)
            likelihood_noemu = get_likelihood(observable=observable_noemu, marg=True, logger=LOGGER)
            LOGGER.info("Likelihood used to perform resampling")
            print_priors(likelihood_noemu)
            if (ntot := sum(_.shape[0] for _ in chains)) > 20000:
                _nthin = ntot // 20000
                LOGGER.info(f"thin chains by a factor of {_nthin}")
                chains = [_[::_nthin] for _ in chains]
            resampler = Resampler(
                likelihood=likelihood_noemu,
                chains=chains,
                save_fn=[_path.parent / ("resample_" + _path.name + f"_{i}.npz") for i in range(len(chains))],
            )
            resampler.run()
        COMM.barrier()

    if "stats" in args.todo:
        # chain
        save_fn = _path.parent / ("stats_chain_" + _path.name + ".txt")
        with log_root_only(mpicomm=COMM):
            load_chains(_path.parent, regex=("chain_" + _path.name + r"_\d+.npz")).to_stats(
                fn=save_fn, params=likelihood.varied_params, tablefmt="pretty"
            )

        # resample
        save_fn = _path.parent / ("stats_resample_" + _path.name + ".txt")
        with log_root_only(mpicomm=COMM):
            try:
                load_chains(_path.parent, regex=("resample_" + _path.name + r"_\d+.npz"), burnin=0).to_stats(
                    fn=save_fn, params=likelihood.varied_params, tablefmt="pretty"
                )
            except FileNotFoundError:
                pass

        # profile
        save_fn = _path.parent / ("stats_profile_" + _path.name + ".txt")
        with log_root_only(mpicomm=COMM):
            Profiles.load(_path.parent / ("profile_" + _path.name + ".npy")).to_stats(
                fn=save_fn, params=likelihood.varied_params, tablefmt="pretty"
            )

        # fit
        save_fn = _path.parent / ("stats_fit_" + _path.name + ".pdf")
        for param in likelihood.all_params:
            if param.derived == ".auto":
                param.update(derived=False)
        LOGGER.info("Likelihood used to evaluate the best-fit plot")
        print_priors(likelihood)
        with log_root_only(mpicomm=COMM):
            bf = load_bestfits(_path.parent / ("profile_" + _path.name + ".npy"), input=True)
        LOGGER.info(f"Loaded bestfit: {bf}")
        likelihood(**bf)
        fig = likelihood.observables[0].plot()
        if COMM.rank == 0:
            try:
                from samples import FitProduct
            except ImportError:
                pass
            else:
                _p = FitProduct(profile=Profiles.load(_path.parent / ("profile_" + _path.name + ".npy")))
                fig.get_axes()[0].set_title(_p.chi2_dof_str())
                fig.savefig(save_fn, bbox_inches="tight")
        COMM.barrier()
