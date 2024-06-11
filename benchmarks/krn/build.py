import argparse
import copy
import dataclasses
import enum
import typing


class System(enum.Enum):
    APOA1 = "APOA1"
    DHFR = "DHFR"
    UUN = "UUN"
    TDP43 = "TDP43"


class YN(enum.Enum):
    YES = "YES"
    NO = "NO"


@dataclasses.dataclass
class BlockFunctions:
    def to_string(self):
        # Get the class name and remove the "BLOCK" part
        name = self.__class__.__name__.upper()
        name = name.replace("BLOCK", "")
        # Confirm that at least one attribute is not None
        if all(value is None for value in dataclasses.asdict(self).values()):
            return ""
        # Create the string representation of the block
        string = f"[{name}]\n"
        for key, value in dataclasses.asdict(self).items():
            # Skip None values
            if value is not None:
                # Convert enum values to their string representation
                if isinstance(value, enum.Enum):
                    value = value.value
                # Add the key-value pair to the string
                string += f"{key} = {value}\n"
        return string


@dataclasses.dataclass
class InputBlock(BlockFunctions):
    topfile: typing.Optional[str] = None
    parfile: typing.Optional[str] = None
    psffile: typing.Optional[str] = None
    pdbfile: typing.Optional[str] = None
    strfile: typing.Optional[str] = None
    rstfile: typing.Optional[str] = None
    prmtopfile: typing.Optional[str] = None
    ambcrdfile: typing.Optional[str] = None
    grotopfile: typing.Optional[str] = None
    grocrdfile: typing.Optional[str] = None


class Forcefield(enum.Enum):
    CHARMM = "CHARMM"
    AMBER = "AMBER"
    RESIDCG = "RESIDCG"


class Electrostatic(enum.Enum):
    PME = "PME"
    CUTOFF = "CUTOFF"


class PMEScheme(enum.Enum):
    OPT_1DALLTOALL = "OPT_1DALLTOALL"
    OPT_2DALLTOALL = "OPT_2DALLTOALL"
    NOOPT_1DALLTOALL = "NOOPT_1DALLTOALL"
    NOOPT_2DALLTOALL = "NOOPT_2DALLTOALL"


@dataclasses.dataclass
class EnergyBlock(BlockFunctions):
    forcefield: typing.Optional[Forcefield] = None
    electrostatic: typing.Optional[Electrostatic] = None
    switchdist: typing.Optional[float] = None
    cutoffdist: typing.Optional[float] = None
    pairlistdist: typing.Optional[float] = None
    pme_alpha: typing.Optional[float] = None
    pme_ngrid_x: typing.Optional[int] = None
    pme_ngrid_y: typing.Optional[int] = None
    pme_ngrid_z: typing.Optional[int] = None
    pme_nspline: typing.Optional[int] = None
    pme_scheme: typing.Optional[PMEScheme] = None
    water_model: typing.Optional[str] = None
    cg_cutoffdist_ele: typing.Optional[float] = None
    cg_cutoffdist_126: typing.Optional[float] = None
    cg_pairlistdist_ele: typing.Optional[float] = None
    cg_pairlistdist_126: typing.Optional[float] = None
    cg_IDR_HPS_epsilon: typing.Optional[float] = None


class Integrator(enum.Enum):
    VVER = "VVER"


@dataclasses.dataclass
class DynamicsBlock(BlockFunctions):
    integrator: typing.Optional[Integrator] = None
    nsteps: typing.Optional[int] = None
    timestep: typing.Optional[float] = None
    eneout_period: typing.Optional[int] = None
    nbupdate_period: typing.Optional[int] = None
    crdout_period: typing.Optional[int] = None
    rstout_period: typing.Optional[int] = None
    iseed: typing.Optional[int] = None


@dataclasses.dataclass
class ConstraintsBlock(BlockFunctions):
    rigid_bond: typing.Optional[YN] = None
    water_model: typing.Optional[str] = None


class Ensemble(enum.Enum):
    NVE = "NVE"
    NVT = "NVT"
    NPT = "NPT"


class TPControl(enum.Enum):
    BUSSI = "BUSSI"
    LANGEVIN = "LANGEVIN"


@dataclasses.dataclass
class EnsembleBlock(BlockFunctions):
    ensemble: typing.Optional[Ensemble] = None
    tpcontrol: typing.Optional[TPControl] = None
    pressure: typing.Optional[float] = None
    temperature: typing.Optional[float] = None
    group_tp: typing.Optional[YN] = None
    gamma_t: typing.Optional[float] = None


class BoundaryType(enum.Enum):
    PBC = "PBC"
    NOBC = "NOBC"


@dataclasses.dataclass
class BoundaryBlock(BlockFunctions):
    type: typing.Optional[BoundaryType] = None
    domain_x: typing.Optional[int] = None
    domain_y: typing.Optional[int] = None
    domain_z: typing.Optional[int] = None


@dataclasses.dataclass
class Configuration:
    input: typing.Optional[InputBlock] = None
    energy: typing.Optional[EnergyBlock] = None
    dynamics: typing.Optional[DynamicsBlock] = None
    constraints: typing.Optional[ConstraintsBlock] = None
    ensemble: typing.Optional[EnsembleBlock] = None
    boundary: typing.Optional[BoundaryBlock] = None
    domains: typing.List[tuple[int, int, int]] = dataclasses.field(default_factory=list)
    job_time: typing.Optional[str] = None
    job_bin: typing.Optional[str] = None

    def to_string(self):
        string = ""
        for field in dataclasses.fields(self):
            block = getattr(self, field.name)
            if isinstance(block, BlockFunctions):
                string += f"{block.to_string()}\n"
        return string

    def validate_pme(self):
        assert self.energy is not None
        assert self.boundary is not None

        scheme = self.energy.pme_scheme
        ngrid_x = self.energy.pme_ngrid_x
        ngrid_y = self.energy.pme_ngrid_y
        ngrid_z = self.energy.pme_ngrid_z
        nspline = self.energy.pme_nspline
        domain_x = self.boundary.domain_x
        domain_y = self.boundary.domain_y
        domain_z = self.boundary.domain_z

        assert scheme is not None
        assert ngrid_x is not None
        assert ngrid_y is not None
        assert ngrid_z is not None
        assert nspline is not None
        assert domain_x is not None
        assert domain_y is not None
        assert domain_z is not None

        if scheme == PMEScheme.OPT_1DALLTOALL or scheme == PMEScheme.NOOPT_1DALLTOALL:
            assert ngrid_x % (2 * domain_x) == 0
            if domain_z % 2 == 0:
                assert ngrid_y % (domain_y * domain_z) == 0
            else:
                assert ngrid_y % (domain_y * domain_z * 2) == 0
            assert ngrid_z % (domain_x * domain_z) == 0
            assert ngrid_z % (domain_y * domain_z) == 0

        if scheme == PMEScheme.OPT_2DALLTOALL or scheme == PMEScheme.NOOPT_2DALLTOALL:
            assert ngrid_x % (2 * domain_x) == 0
            if domain_z % 2 == 0:
                assert ngrid_y % (domain_y * domain_z) == 0
            else:
                assert ngrid_y % (domain_y * domain_z * 2) == 0
            assert ngrid_z % (domain_x * domain_z) == 0

        if scheme == PMEScheme.OPT_1DALLTOALL or scheme == PMEScheme.OPT_2DALLTOALL:
            assert ngrid_x // domain_x > nspline
            assert ngrid_y // domain_y > nspline
            assert ngrid_z // domain_z > nspline


class Compiler(enum.Enum):
    GNU = "GNU"
    CRAY = "CRAY"


class Job:
    def __init__(
        self,
        inp_idx: int,
        job_idx: int,
        nodes: int,
        ntasks_per_node: int,
        cpus_per_task: int,
        gpus_per_node: int,
        compiler: Compiler,
        job_time: str,
        job_bin: str,
        nruns: int = 5,
    ):
        self.inp_idx = inp_idx
        self.job_idx = job_idx
        self.nodes = nodes
        self.ntasks_per_node = ntasks_per_node
        self.cpus_per_task = cpus_per_task
        self.gpus_per_node = gpus_per_node
        self.compiler = compiler
        self.job_time = job_time
        self.job_bin = job_bin
        self.nruns = nruns

    def to_string(self):
        string = ""
        string += "#!/bin/bash -e\n"
        string += f"#SBATCH --job-name={self.inp_idx:04}-{self.job_idx:04}\n"
        string += "#SBATCH --account=Project_462000123\n"
        string += f"#SBATCH --time={self.job_time}\n"
        string += "#SBATCH --partition=standard-g\n"
        string += "#SBATCH --mem=0\n"
        string += f"#SBATCH --nodes={self.nodes}\n"
        string += f"#SBATCH --ntasks-per-node={self.ntasks_per_node}\n"
        string += f"#SBATCH --cpus-per-task={self.cpus_per_task}\n"
        string += f"#SBATCH --gpus-per-node={self.gpus_per_node}\n"
        string += "#SBATCH --exclusive\n"
        string += "#SBATCH -o %x-%j.out\n"
        string += "export PMI_NO_PREINITIALIZE=y\n"
        if self.compiler == Compiler.GNU:
            string += "module load PrgEnv-gnu\n"
        elif self.compiler == Compiler.CRAY:
            string += "module load PrgEnv-cray\n"
        string += "module load craype-x86-trento\n"
        string += "module load craype-accel-amd-gfx90a\n"
        string += "module load CrayEnv\n"
        string += "module load rocm/5.6.1\n"
        string += "module load cray-libsci\n"
        string += "export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n"
        string += "export OMP_PROC_BIND=true\n"
        string += "export OMP_PLACES=cores\n"
        string += 'CPU_BIND="mask_cpu:"\n'
        if self.ntasks_per_node == 8:
            string += 'CPU_BIND="${CPU_BIND}0x00fe000000000000,"\n'
            string += 'CPU_BIND="${CPU_BIND}0xfe00000000000000,"\n'
            string += 'CPU_BIND="${CPU_BIND}0x0000000000fe0000,"\n'
            string += 'CPU_BIND="${CPU_BIND}0x00000000fe000000,"\n'
            string += 'CPU_BIND="${CPU_BIND}0x00000000000000fe,"\n'
            string += 'CPU_BIND="${CPU_BIND}0x000000000000fe00,"\n'
            string += 'CPU_BIND="${CPU_BIND}0x000000fe00000000,"\n'
            string += 'CPU_BIND="${CPU_BIND}0x0000fe0000000000"\n'
        elif self.ntasks_per_node == 4:
            string += 'CPU_BIND="${CPU_BIND}0xfefe000000000000,"\n'
            string += 'CPU_BIND="${CPU_BIND}0x00000000fefe0000,"\n'
            string += 'CPU_BIND="${CPU_BIND}0x000000000000fefe,"\n'
            string += 'CPU_BIND="${CPU_BIND}0x0000fefe00000000"\n'
        elif self.ntasks_per_node == 2:
            string += 'CPU_BIND="${CPU_BIND}0xfefe0000fefe0000,"\n'
            string += 'CPU_BIND="${CPU_BIND}0x0000fefe0000fefe"\n'
        elif self.ntasks_per_node == 1:
            string += 'CPU_BIND="${CPU_BIND}0xfefefefefefefefe"\n'
        string += 'export SLURM_CPU_BIND="${CPU_BIND}"\n'
        string += "\n"
        string += "# Warm up\n"
        string += f"srun {self.job_bin} {self.inp_idx:04}.INP\n"
        string += "\n"
        string += "# Benchmark\n"
        for i in range(self.nruns):
            prefix = f"{self.inp_idx:04}-{self.job_idx:04}-{self.nodes:04}-{self.ntasks_per_node:04}-{self.cpus_per_task:04}-{self.gpus_per_node:04}-{i:04}"
            stdout = f"{prefix}.OUT"
            stderr = f"{prefix}.ERR"

            string += f"srun {self.job_bin} {self.inp_idx:04}.INP"
            string += f" > {stdout}"
            string += f" 2> {stderr}"
            string += "\n"

        return string


SYSTEM_CONFIGURATIONS = {
    System.APOA1: Configuration(
        input=InputBlock(
            topfile="../inputs/apoa1/top_all27_prot_lipid.rtf",
            parfile="../inputs/apoa1/par_all27_prot_lipid.prm",
            psffile="../inputs/apoa1/apoa1.psf",
            pdbfile="../inputs/apoa1/apoa1.pdb",
            rstfile="../inputs/apoa1/apoa1.rst",
        ),
        energy=EnergyBlock(
            forcefield=Forcefield.CHARMM,
            electrostatic=Electrostatic.PME,
            switchdist=10.0,
            cutoffdist=12.0,
            pairlistdist=13.5,
            pme_alpha=0.34,
            pme_ngrid_x=80,
            pme_ngrid_y=80,
            pme_ngrid_z=64,
            pme_nspline=4,
        ),
        dynamics=DynamicsBlock(
            integrator=Integrator.VVER,
            nsteps=10000,
            timestep=0.002,
            eneout_period=100,
            nbupdate_period=10,
        ),
        constraints=ConstraintsBlock(
            rigid_bond=YN.YES,
        ),
        ensemble=EnsembleBlock(
            ensemble=Ensemble.NPT,
            tpcontrol=TPControl.BUSSI,
            pressure=1.0,
            temperature=300.0,
            group_tp=YN.YES,
        ),
        boundary=BoundaryBlock(type=BoundaryType.PBC),
        domains=[
            # 4
            (2, 2, 1),
            # 8
            (2, 2, 2),
            # 16
            (4, 2, 2),
            (2, 4, 2),
            (2, 2, 4),
            # 32
            (2, 4, 4),
            (4, 2, 4),
            (4, 4, 2),
            # 64
            (4, 4, 4),
        ],
        job_time="01:00:00",
        job_bin="$(pwd)/../../src/spdyn/spdyn",
    ),
    System.DHFR: Configuration(
        input=InputBlock(
            prmtopfile="../inputs/jac_amber/step3_input.parm7",
            ambcrdfile="../inputs/jac_amber/step3_input.rst7",
            rstfile="../inputs/jac_amber/equil.rst",
        ),
        energy=EnergyBlock(
            forcefield=Forcefield.AMBER,
            electrostatic=Electrostatic.PME,
            switchdist=8.0,
            cutoffdist=8.0,
            pairlistdist=9.5,
            pme_ngrid_x=48,
            pme_ngrid_y=48,
            pme_ngrid_z=48,
            pme_nspline=4,
        ),
        dynamics=DynamicsBlock(
            integrator=Integrator.VVER,
            nsteps=10000,
            timestep=0.002,
            eneout_period=100,
            nbupdate_period=10,
        ),
        constraints=ConstraintsBlock(
            rigid_bond=YN.YES,
            water_model="WAT",
        ),
        ensemble=EnsembleBlock(
            ensemble=Ensemble.NPT,
            tpcontrol=TPControl.BUSSI,
            pressure=1.0,
            temperature=300.0,
            group_tp=YN.YES,
        ),
        boundary=BoundaryBlock(type=BoundaryType.PBC),
        domains=[
            # 4
            (2, 2, 1),
            # 8
            (2, 2, 2),
            # 16
            (4, 2, 2),
            (2, 4, 2),
            (2, 2, 4),
            # 32
            (2, 4, 4),
            (4, 2, 4),
            (4, 4, 2),
            # 64
            (4, 4, 4),
        ],
        job_time="01:00:00",
        job_bin="$(pwd)/../../src/spdyn/spdyn",
    ),
    System.UUN: Configuration(
        input=InputBlock(
            topfile="../inputs/uun/toppar/top_all36_prot.rtf,../inputs/uun/toppar/top_all36_na.rtf,../inputs/uun/toppar/top_all36_lipid.rtf,../inputs/uun/toppar/top_all36_cgenff.rtf,../inputs/uun/toppar/top_all36_carb.rtf",
            parfile="../inputs/uun/toppar/par_all36_prot.prm,../inputs/uun/toppar/par_all36_na.prm,../inputs/uun/toppar/par_all36_carb.prm,../inputs/uun/toppar/par_all36_lipid.prm,../inputs/uun/toppar/par_all36_cgenff.prm",
            strfile="../inputs/uun/toppar/toppar_water_ions.genesis.str",
            psffile="../inputs/uun/uun.psf",
            pdbfile="../inputs/uun/uun.pdb",
            rstfile="../inputs/uun/uun.rst",
        ),
        energy=EnergyBlock(
            forcefield=Forcefield.CHARMM,
            electrostatic=Electrostatic.PME,
            switchdist=10.0,
            cutoffdist=12.0,
            pairlistdist=13.5,
            pme_ngrid_x=128,
            pme_ngrid_y=128,
            pme_ngrid_z=128,
            pme_nspline=4,
            water_model="NONE",
        ),
        dynamics=DynamicsBlock(
            integrator=Integrator.VVER,
            nsteps=10000,
            timestep=0.002,
            eneout_period=100,
            nbupdate_period=10,
        ),
        constraints=ConstraintsBlock(
            rigid_bond=YN.YES,
        ),
        ensemble=EnsembleBlock(
            ensemble=Ensemble.NPT,
            tpcontrol=TPControl.BUSSI,
            temperature=300.0,
            group_tp=YN.YES,
        ),
        boundary=BoundaryBlock(type=BoundaryType.PBC),
        domains=[
            # 4
            (2, 2, 1),
            # 8
            (2, 2, 2),
            # 16
            (4, 2, 2),
            (2, 4, 2),
            (2, 2, 4),
            # 32
            (2, 4, 4),
            (4, 2, 4),
            (4, 4, 2),
            # 64
            (4, 4, 4),
            # 128
            (8, 4, 4),
            (4, 8, 4),
            (4, 4, 8),
            # 256
            (4, 8, 8),
            (8, 4, 8),
            (8, 8, 4),
            # 512
            (8, 8, 8),
        ],
        job_time="03:00:00",
        job_bin="$(pwd)/../../src/spdyn/spdyn",
    ),
    System.TDP43: Configuration(
        input=InputBlock(
            grotopfile="../rCG_Droplets_50_50_50_100_b100_f52_g20_d0_rho0.62.top",
            grocrdfile="../crd/rCG_Droplets_50_50_50_100_b100_f52_g20_d0_rho0.62.gro",
            rstfile="../_DROPLET_test_eq_s_1_d_2_01_02.rst",
        ),
        energy=EnergyBlock(
            forcefield=Forcefield.RESIDCG,
            electrostatic=Electrostatic.CUTOFF,
            cg_cutoffdist_ele=35,
            cg_cutoffdist_126=20,
            cg_pairlistdist_ele=38,
            cg_pairlistdist_126=23,
            cg_IDR_HPS_epsilon=0.2,
        ),
        dynamics=DynamicsBlock(
            integrator=Integrator.VVER,
            nsteps=10000,
            timestep=0.010,
            eneout_period=1000,
            nbupdate_period=20,
        ),
        constraints=ConstraintsBlock(
            rigid_bond=YN.NO,
        ),
        ensemble=EnsembleBlock(
            ensemble=Ensemble.NVT,
            tpcontrol=TPControl.LANGEVIN,
            temperature=290.0,
            gamma_t=0.01,
        ),
        boundary=BoundaryBlock(type=BoundaryType.PBC),
        domains=[
            # 8
            (2, 2, 2),
            # 16
            (4, 2, 2),
            # 32
            (4, 4, 2),
            # 64
            (4, 4, 4),
            # 128
            (8, 4, 4),
            # 256
            (8, 8, 4),
            # 512
            (8, 8, 8),
            # 1024
            (16, 8, 8),
            # 2048
            (16, 16, 8),
            # 4096
            (16, 16, 16),
            # 8192
            (32, 16, 16),
        ],
        job_time="00:30:00",
        job_bin="./cgdyn",
    ),
}


PARALLEL = {
    4: [
        (1, 4, 14, 4),
        (2, 2, 28, 2),
        (4, 1, 56, 1),
    ],
    8: [
        (1, 8, 7, 8),
        (2, 4, 14, 4),
        (4, 2, 28, 2),
        (8, 1, 56, 1),
    ],
    16: [
        (2, 8, 7, 8),
        (4, 4, 14, 4),
        (8, 2, 28, 2),
        (16, 1, 56, 1),
    ],
    32: [
        (4, 8, 7, 8),
        (8, 4, 14, 4),
        (16, 2, 28, 2),
        (32, 1, 56, 1),
    ],
    64: [
        (8, 8, 7, 8),
        (16, 4, 14, 4),
        (32, 2, 28, 2),
        (64, 1, 56, 1),
    ],
    128: [
        (16, 8, 7, 8),
        (32, 4, 14, 4),
        (64, 2, 28, 2),
        (128, 1, 56, 1),
    ],
    256: [
        (32, 8, 7, 8),
        (64, 4, 14, 4),
        (128, 2, 28, 2),
        (256, 1, 56, 1),
    ],
    512: [
        (64, 8, 7, 8),
        (128, 4, 14, 4),
        (256, 2, 28, 2),
        (512, 1, 56, 1),
    ],
    1024: [
        (128, 8, 7, 8),
        (256, 4, 14, 4),
        (512, 2, 28, 2),
        (1024, 1, 56, 1),
    ],
    2048: [
        (256, 8, 7, 8),
        (512, 4, 14, 4),
        (1024, 2, 28, 2),
        # (2048, 1, 56, 1),
    ],
    4096: [
        (512, 8, 7, 8),
        (1024, 4, 14, 4),
        # (2048, 2, 28, 2),
        # (4096, 1, 56, 1),
    ],
    8192: [
        (1024, 8, 7, 8),
        # (2048, 4, 14, 4),
        # (4096, 2, 28, 2),
        # (8192, 1, 56, 1),
    ],
    # 16384: [
    #    (2048, 8, 7, 8),
    #    (4096, 4, 14, 4),
    #    (8192, 2, 28, 2),
    #    (16384, 1, 56, 1),
    # ],
    # 32768: [
    #    (4096, 8, 7, 8),
    #    (8192, 4, 14, 4),
    #    (16384, 2, 28, 2),
    #    (32768, 1, 56, 1),
    # ],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compiler", type=str, default="GNU", help="Compiler to use (GNU, CRAY)"
    )
    parser.add_argument(
        "--use-pme", action="store_true", help="Use PME scheme variations"
    )
    parser.add_argument(
        "--system",
        type=str,
        default="APOA1",
        help="System to use (APOA1, DHFR, UUN, TDP43)",
    )
    args = parser.parse_args()

    compiler = Compiler.GNU if args.compiler == "GNU" else Compiler.CRAY

    system_choice = System[args.system]
    base_configuration = SYSTEM_CONFIGURATIONS[system_choice]

    configurations_stage1: list[Configuration] = [copy.deepcopy(base_configuration)]
    configurations_stage2: list[Configuration] = []
    configurations_stage3: list[Configuration] = []

    use_pme = args.use_pme
    if use_pme:
        for configuration in configurations_stage1:
            for pme_scheme in PMEScheme:
                new_configuration = copy.deepcopy(configuration)
                assert new_configuration.energy is not None
                new_configuration.energy.pme_scheme = pme_scheme
                configurations_stage2.append(new_configuration)
    else:
        configurations_stage2 = copy.deepcopy(configurations_stage1)

    for configuration in configurations_stage2:
        for domain in configuration.domains:
            new_configuration = copy.deepcopy(configuration)
            assert new_configuration.boundary is not None
            (
                new_configuration.boundary.domain_x,
                new_configuration.boundary.domain_y,
                new_configuration.boundary.domain_z,
            ) = domain
            configurations_stage3.append(new_configuration)

    configurations = copy.deepcopy(configurations_stage3)

    for cfg_idx, configuration in enumerate(configurations):
        assert configuration.job_time is not None
        assert configuration.job_bin is not None

        with open(f"{cfg_idx:04}.INP", "w") as file:
            file.write(configuration.to_string())

        boundary = configuration.boundary
        assert boundary is not None

        domain_x = boundary.domain_x
        domain_y = boundary.domain_y
        domain_z = boundary.domain_z
        assert domain_x is not None
        assert domain_y is not None
        assert domain_z is not None

        ndomains = domain_x * domain_y * domain_z
        parallel = PARALLEL[ndomains]

        jobs: list[Job] = []
        for job_idx, (
            nodes,
            ntasks_per_node,
            cpus_per_task,
            gpus_per_node,
        ) in enumerate(parallel):
            jobs.append(
                Job(
                    cfg_idx,
                    job_idx,
                    nodes,
                    ntasks_per_node,
                    cpus_per_task,
                    gpus_per_node,
                    compiler,
                    configuration.job_time,
                    configuration.job_bin,
                )
            )

        for job in jobs:
            with open(f"{cfg_idx:04}-{job.job_idx:04}.LUMI.SLURM", "w") as file:
                file.write(job.to_string())


if __name__ == "__main__":
    main()
