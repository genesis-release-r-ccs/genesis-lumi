import argparse
import copy
import dataclasses
import enum
import typing


@dataclasses.dataclass
class BlockFunctions:
    def to_string(self):
        # Get the class name and remove the "BLOCK" part
        name = self.__class__.__name__.upper()
        name = name.replace("BLOCK", "")
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


INPUTS = {
    "APOA1": InputBlock(
        topfile="../inputs/apoa1/top_all27_prot_lipid.rtf",
        parfile="../inputs/apoa1/par_all27_prot_lipid.prm",
        psffile="../inputs/apoa1/apoa1.psf",
        pdbfile="../inputs/apoa1/apoa1.pdb",
        rstfile="../inputs/apoa1/apoa1.rst",
    ),
    "DHFR": InputBlock(
        prmtopfile="../inputs/jac_amber/step3_input.parm7",
        ambcrdfile="../inputs/jac_amber/step3_input.rst7",
        rstfile="../inputs/jac_amber/equil.rst",
    ),
    "UUN": InputBlock(
        topfile="../inputs/uun/toppar/top_all36_prot.rtf,../inputs/uun/toppar/top_all36_na.rtf,../inputs/uun/toppar/top_all36_lipid.rtf,../inputs/uun/toppar/top_all36_cgenff.rtf,../inputs/uun/toppar/top_all36_carb.rtf",
        parfile="../inputs/uun/toppar/par_all36_prot.prm,../inputs/uun/toppar/par_all36_na.prm,../inputs/uun/toppar/par_all36_carb.prm,../inputs/uun/toppar/par_all36_lipid.prm,../inputs/uun/toppar/par_all36_cgenff.prm",
        strfile="../inputs/uun/toppar/toppar_water_ions.genesis.str",
        psffile="../inputs/uun/uun.psf",
        pdbfile="../inputs/uun/uun.pdb",
        rstfile="../inputs/uun/uun.rst",
    ),
}


class PMEScheme(enum.Enum):
    OPT_1DALLTOALL = "OPT_1DALLTOALL"
    OPT_2DALLTOALL = "OPT_2DALLTOALL"
    NOOPT_1DALLTOALL = "NOOPT_1DALLTOALL"
    NOOPT_2DALLTOALL = "NOOPT_2DALLTOALL"


@dataclasses.dataclass
class EnergyBlock(BlockFunctions):
    forcefield: typing.Optional[str] = None
    electrostatic: typing.Optional[str] = None
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


ENERGIES = {
    "APOA1": EnergyBlock(
        forcefield="CHARMM",
        electrostatic="PME",
        switchdist=10.0,
        cutoffdist=12.0,
        pairlistdist=13.5,
        pme_alpha=0.34,
        pme_ngrid_x=80,
        pme_ngrid_y=80,
        pme_ngrid_z=64,
        pme_nspline=4,
    ),
    "DHFR": EnergyBlock(
        forcefield="AMBER",
        electrostatic="PME",
        switchdist=8.0,
        cutoffdist=8.0,
        pairlistdist=9.5,
        pme_ngrid_x=48,
        pme_ngrid_y=48,
        pme_ngrid_z=48,
        pme_nspline=4,
    ),
    "UUN": EnergyBlock(
        forcefield="CHARMM",
        electrostatic="PME",
        switchdist=10.0,
        cutoffdist=12.0,
        pairlistdist=13.5,
        pme_ngrid_x=128,
        pme_ngrid_y=128,
        pme_ngrid_z=128,
        pme_nspline=4,
        water_model="NONE",
    ),
}


@dataclasses.dataclass
class DynamicsBlock(BlockFunctions):
    integrator: typing.Optional[str] = None
    nsteps: typing.Optional[int] = None
    timestep: typing.Optional[float] = None
    eneout_period: typing.Optional[int] = None
    nbupdate_period: typing.Optional[int] = None


DYNAMICS = {
    "APOA1": DynamicsBlock(
        integrator="VVER",
        nsteps=10000,
        timestep=0.002,
        eneout_period=100,
        nbupdate_period=10,
    ),
    "DHFR": DynamicsBlock(
        integrator="VVER",
        nsteps=10000,
        timestep=0.002,
        eneout_period=100,
        nbupdate_period=10,
    ),
    "UUN": DynamicsBlock(
        integrator="VVER",
        nsteps=10000,
        timestep=0.002,
        eneout_period=100,
        nbupdate_period=10,
    ),
}


@dataclasses.dataclass
class ConstraintsBlock(BlockFunctions):
    rigid_bond: typing.Optional[str] = None
    water_model: typing.Optional[str] = None


CONSTRAINTS = {
    "APOA1": ConstraintsBlock(
        rigid_bond="YES",
    ),
    "DHFR": ConstraintsBlock(
        rigid_bond="YES",
        water_model="WAT",
    ),
    "UUN": ConstraintsBlock(
        rigid_bond="YES",
    ),
}


@dataclasses.dataclass
class EnsembleBlock(BlockFunctions):
    ensemble: typing.Optional[str] = None
    tpcontrol: typing.Optional[str] = None
    pressure: typing.Optional[float] = None
    temperature: typing.Optional[float] = None
    group_tp: typing.Optional[str] = None


ENSEMBLES = {
    "APOA1": EnsembleBlock(
        ensemble="NPT",
        tpcontrol="BUSSI",
        pressure=1.0,
        temperature=300.0,
        group_tp="YES",
    ),
    "DHFR": EnsembleBlock(
        ensemble="NPT",
        tpcontrol="BUSSI",
        pressure=1.0,
        temperature=300.0,
        group_tp="YES",
    ),
    "UUN": EnsembleBlock(
        ensemble="NPT",
        tpcontrol="BUSSI",
        temperature=300.0,
        group_tp="YES",
    ),
}


@dataclasses.dataclass
class BoundaryBlock(BlockFunctions):
    type: typing.Optional[str] = None
    domain_x: typing.Optional[int] = None
    domain_y: typing.Optional[int] = None
    domain_z: typing.Optional[int] = None


BOUNDARIES = {
    "APOA1": BoundaryBlock(type="PBC"),
    "DHFR": BoundaryBlock(type="PBC"),
    "UUN": BoundaryBlock(type="PBC"),
}

DOMAINS = {
    "APOA1": [
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
    "DHFR": [
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
    "UUN": [
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
}


@dataclasses.dataclass
class Configuration:
    input: InputBlock
    energy: EnergyBlock
    dynamics: DynamicsBlock
    constraints: ConstraintsBlock
    ensemble: EnsembleBlock
    boundary: BoundaryBlock

    def to_string(self):
        input = self.input.to_string()
        energy = self.energy.to_string()
        dynamics = self.dynamics.to_string()
        constraints = self.constraints.to_string()
        ensemble = self.ensemble.to_string()
        boundary = self.boundary.to_string()
        string = f"{input}\n{energy}\n{dynamics}\n{constraints}\n{ensemble}\n{boundary}"
        return string

    def validate_pme(self):
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


SYSTEMS = ["APOA1", "DHFR", "UUN"]


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
        nruns: int = 5,
    ):
        self.inp_idx = inp_idx
        self.job_idx = job_idx
        self.nodes = nodes
        self.ntasks_per_node = ntasks_per_node
        self.cpus_per_task = cpus_per_task
        self.gpus_per_node = gpus_per_node
        self.compiler = compiler
        self.nruns = nruns

    def to_string(self):
        string = ""
        string += "#!/bin/bash -e\n"
        string += f"#SBATCH --job-name={self.inp_idx:04}-{self.job_idx:04}\n"
        string += "#SBATCH --account=Project_462000123\n"
        string += "#SBATCH --time=03:00:00\n"
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
        string += f'srun "$(pwd)/../../src/spdyn/spdyn" {self.inp_idx:04}.INP\n'
        string += "\n"
        string += "# Benchmark\n"
        for i in range(self.nruns):
            string += f'srun "$(pwd)/../../src/spdyn/spdyn" {self.inp_idx:04}.INP 2>&1 | tee {self.inp_idx:04}-{self.job_idx:04}-{self.nodes:04}-{self.ntasks_per_node:04}-{self.cpus_per_task:04}-{self.gpus_per_node:04}-{i:04}.OUT\n'

        return string


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
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compiler", type=str, default="GNU", help="Compiler to use (GNU, CRAY)"
    )
    args = parser.parse_args()

    compiler = Compiler.GNU if args.compiler == "GNU" else Compiler.CRAY

    configurations: list[Configuration] = []
    for system in SYSTEMS:
        input = copy.deepcopy(INPUTS[system])
        dynamics = copy.deepcopy(DYNAMICS[system])
        constraints = copy.deepcopy(CONSTRAINTS[system])
        ensemble = copy.deepcopy(ENSEMBLES[system])
        for pme_scheme in PMEScheme:
            energy = copy.deepcopy(ENERGIES[system])
            energy.pme_scheme = pme_scheme
            for domain in DOMAINS[system]:
                boundary = copy.deepcopy(BOUNDARIES[system])
                boundary.domain_x, boundary.domain_y, boundary.domain_z = domain
                configuration = Configuration(
                    input,
                    energy,
                    dynamics,
                    constraints,
                    ensemble,
                    boundary,
                )
                configuration.validate_pme()
                configurations.append(configuration)

    for inp_idx, configuration in enumerate(configurations):
        with open(f"{inp_idx:04}.INP", "w") as file:
            file.write(configuration.to_string())

        domain_x = configuration.boundary.domain_x
        domain_y = configuration.boundary.domain_y
        domain_z = configuration.boundary.domain_z

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
                    inp_idx,
                    job_idx,
                    nodes,
                    ntasks_per_node,
                    cpus_per_task,
                    gpus_per_node,
                    compiler,
                )
            )

        for job in jobs:
            with open(f"{inp_idx:04}-{job.job_idx:04}.LUMI.SLURM", "w") as file:
                file.write(job.to_string())


if __name__ == "__main__":
    main()
