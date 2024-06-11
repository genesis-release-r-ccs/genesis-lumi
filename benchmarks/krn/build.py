import argparse
import copy
import dataclasses
import enum
import typing

import tomllib


class YesNo(enum.Enum):
    YES = "YES"
    NO = "NO"


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


class NonbondKernel(enum.Enum):
    FUGAKU = "FUGAKU"
    INTEL = "INTEL"
    GENERIC = "GENERIC"


class Integrator(enum.Enum):
    VVER = "VVER"


class Ensemble(enum.Enum):
    NVE = "NVE"
    NVT = "NVT"
    NPT = "NPT"


class TPControl(enum.Enum):
    BUSSI = "BUSSI"
    LANGEVIN = "LANGEVIN"


class BoundaryType(enum.Enum):
    PBC = "PBC"
    NOBC = "NOBC"


class Compiler(enum.Enum):
    GNU = "GNU"
    CRAY = "CRAY"


@dataclasses.dataclass
class Block:
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
class InputBlock(Block):
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


@dataclasses.dataclass
class EnergyBlock(Block):
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
    nonbond_kernel: typing.Optional[NonbondKernel] = None
    water_model: typing.Optional[str] = None
    cg_cutoffdist_ele: typing.Optional[float] = None
    cg_cutoffdist_126: typing.Optional[float] = None
    cg_pairlistdist_ele: typing.Optional[float] = None
    cg_pairlistdist_126: typing.Optional[float] = None
    cg_IDR_HPS_epsilon: typing.Optional[float] = None


@dataclasses.dataclass
class DynamicsBlock(Block):
    integrator: typing.Optional[Integrator] = None
    nsteps: typing.Optional[int] = None
    timestep: typing.Optional[float] = None
    eneout_period: typing.Optional[int] = None
    nbupdate_period: typing.Optional[int] = None
    crdout_period: typing.Optional[int] = None
    rstout_period: typing.Optional[int] = None
    iseed: typing.Optional[int] = None


@dataclasses.dataclass
class ConstraintsBlock(Block):
    rigid_bond: typing.Optional[YesNo] = None
    water_model: typing.Optional[str] = None


@dataclasses.dataclass
class EnsembleBlock(Block):
    ensemble: typing.Optional[Ensemble] = None
    tpcontrol: typing.Optional[TPControl] = None
    pressure: typing.Optional[float] = None
    temperature: typing.Optional[float] = None
    group_tp: typing.Optional[YesNo] = None
    gamma_t: typing.Optional[float] = None


@dataclasses.dataclass
class BoundaryBlock(Block):
    type: typing.Optional[BoundaryType] = None
    domain_x: typing.Optional[int] = None
    domain_y: typing.Optional[int] = None
    domain_z: typing.Optional[int] = None


@dataclasses.dataclass
class Configuration:
    input: InputBlock = dataclasses.field(default_factory=InputBlock)
    energy: EnergyBlock = dataclasses.field(default_factory=EnergyBlock)
    dynamics: DynamicsBlock = dataclasses.field(default_factory=DynamicsBlock)
    constraints: ConstraintsBlock = dataclasses.field(default_factory=ConstraintsBlock)
    ensemble: EnsembleBlock = dataclasses.field(default_factory=EnsembleBlock)
    boundary: BoundaryBlock = dataclasses.field(default_factory=BoundaryBlock)

    def to_string(self):
        string = ""
        for field in dataclasses.fields(self):
            item = getattr(self, field.name)
            if isinstance(item, Block):
                string += f"{item.to_string()}\n"
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


@dataclasses.dataclass
class Job:
    max_cpus: typing.Optional[int] = None
    compiler: typing.Optional[Compiler] = None
    time: typing.Optional[str] = None
    bin: typing.Optional[str] = None
    inp_name: typing.Optional[str] = None
    job_name: typing.Optional[str] = None
    nodes: typing.Optional[int] = None
    ntasks_per_node: typing.Optional[int] = None
    cpus_per_task: typing.Optional[int] = None
    gpus_per_node: typing.Optional[int] = None

    def to_string(self):
        assert self.inp_name is not None
        assert self.job_name is not None
        assert self.nodes is not None
        assert self.ntasks_per_node is not None
        assert self.cpus_per_task is not None
        assert self.gpus_per_node is not None
        assert self.compiler is not None
        assert self.time is not None
        assert self.bin is not None

        string = ""
        string += "#!/bin/bash -e\n"
        string += f"#SBATCH --job-name={self.job_name}\n"
        string += "#SBATCH --account=Project_462000123\n"
        string += f"#SBATCH --time={self.time}\n"
        string += "#SBATCH --partition=standard-g\n"
        string += "#SBATCH --mem=0\n"
        string += f"#SBATCH --nodes={self.nodes}\n"
        string += f"#SBATCH --ntasks-per-node={self.ntasks_per_node}\n"
        string += f"#SBATCH --cpus-per-task={self.cpus_per_task}\n"
        string += f"#SBATCH --gpus-per-node={self.gpus_per_node}\n"
        string += "#SBATCH --exclusive\n"
        string += f"#SBATCH -o {self.job_name}.LOG\n"
        string += "export PMI_NO_PREINITIALIZE=y\n"
        match self.compiler:
            case Compiler.GNU:
                string += "module load PrgEnv-gnu\n"
            case Compiler.CRAY:
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
        match self.ntasks_per_node:
            case 8:
                string += 'CPU_BIND="${CPU_BIND}0x00fe000000000000,"\n'
                string += 'CPU_BIND="${CPU_BIND}0xfe00000000000000,"\n'
                string += 'CPU_BIND="${CPU_BIND}0x0000000000fe0000,"\n'
                string += 'CPU_BIND="${CPU_BIND}0x00000000fe000000,"\n'
                string += 'CPU_BIND="${CPU_BIND}0x00000000000000fe,"\n'
                string += 'CPU_BIND="${CPU_BIND}0x000000000000fe00,"\n'
                string += 'CPU_BIND="${CPU_BIND}0x000000fe00000000,"\n'
                string += 'CPU_BIND="${CPU_BIND}0x0000fe0000000000"\n'
            case 4:
                string += 'CPU_BIND="${CPU_BIND}0xfefe000000000000,"\n'
                string += 'CPU_BIND="${CPU_BIND}0x00000000fefe0000,"\n'
                string += 'CPU_BIND="${CPU_BIND}0x000000000000fefe,"\n'
                string += 'CPU_BIND="${CPU_BIND}0x0000fefe00000000"\n'
            case 2:
                string += 'CPU_BIND="${CPU_BIND}0xfefe0000fefe0000,"\n'
                string += 'CPU_BIND="${CPU_BIND}0x0000fefe0000fefe"\n'
            case 1:
                string += 'CPU_BIND="${CPU_BIND}0xfefefefefefefefe"\n'
            case _:
                raise ValueError("Invalid number of tasks per node")
        string += 'export SLURM_CPU_BIND="${CPU_BIND}"\n'
        string += "\n"
        string += "# Warm up\n"
        string += f"srun {self.bin} {self.inp_name}.INP\n"
        string += "\n"
        string += "# Benchmark\n"
        for i in range(3):
            prefix = f"{self.job_name}-{self.nodes:04}-{self.ntasks_per_node:04}-{self.cpus_per_task:04}-{self.gpus_per_node:04}-{i:04}"
            stdout = f"{prefix}.OUT"
            stderr = f"{prefix}.ERR"

            string += f"srun {self.bin} {self.inp_name}.INP"
            string += f" 1> {stdout}"
            string += f" 2> {stderr}"
            string += "\n"

        return string


def build_configurations(toml: dict[str, typing.Any]) -> list[Configuration]:
    # Stage 1
    configuration_stage1 = Configuration()

    # Stage 2
    new_configuration = copy.deepcopy(configuration_stage1)
    input = toml["input"]
    input = {k: v for k, v in input.items() if not isinstance(v, list)}
    new_configuration.input = InputBlock(**input)

    energy = toml["energy"]
    energy = {k: v for k, v in energy.items() if not isinstance(v, list)}
    new_configuration.energy = EnergyBlock(**energy)

    dynamics = toml["dynamics"]
    dynamics = {k: v for k, v in dynamics.items() if not isinstance(v, list)}
    new_configuration.dynamics = DynamicsBlock(**dynamics)

    constraints = toml["constraints"]
    constraints = {k: v for k, v in constraints.items() if not isinstance(v, list)}
    new_configuration.constraints = ConstraintsBlock(**constraints)

    ensemble = toml["ensemble"]
    ensemble = {k: v for k, v in ensemble.items() if not isinstance(v, list)}
    new_configuration.ensemble = EnsembleBlock(**ensemble)

    boundary = toml["boundary"]
    boundary = {k: v for k, v in boundary.items() if not isinstance(v, list)}
    new_configuration.boundary = BoundaryBlock(**boundary)

    configuration_stage2 = copy.deepcopy(new_configuration)

    # Stage 3
    if "pme_scheme" in toml["energy"]:
        new_configurations: list[Configuration] = []
        for pme_scheme in toml["energy"]["pme_scheme"]:
            new_configuration = copy.deepcopy(configuration_stage2)
            new_configuration.energy.pme_scheme = PMEScheme(pme_scheme)
            new_configurations.append(new_configuration)
    else:
        new_configurations = [copy.deepcopy(configuration_stage2)]

    configuration_stage3 = copy.deepcopy(new_configurations)

    # Stage 4
    if "nonbond_kernel" in toml["energy"]:
        new_configurations: list[Configuration] = []
        for nonbond_kernel in toml["energy"]["nonbond_kernel"]:
            for configuration in configuration_stage3:
                new_configuration = copy.deepcopy(configuration)
                new_configuration.energy.nonbond_kernel = NonbondKernel(nonbond_kernel)
                new_configurations.append(new_configuration)
    else:
        new_configurations = copy.deepcopy(configuration_stage3)

    configuration_stage4 = copy.deepcopy(new_configurations)

    # Stage 5
    new_configurations: list[Configuration] = []
    for domain in toml["boundary"]["domains"]:
        for configuration in configuration_stage4:
            new_configuration = copy.deepcopy(configuration)
            new_configuration.boundary.domain_x = domain[0]
            new_configuration.boundary.domain_y = domain[1]
            new_configuration.boundary.domain_z = domain[2]
            new_configurations.append(new_configuration)

    configuration_stage5 = copy.deepcopy(new_configurations)

    # Return the final configurations
    configurations = configuration_stage5
    return configurations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT", type=str, help="Input file in TOML format")
    args = parser.parse_args()

    # Open toml file
    with open(args.INPUT, "rb") as file:
        toml = tomllib.load(file)

    # Build configurations
    configurations = build_configurations(toml)

    print(toml)
    print(configurations)


#    compiler = Compiler.GNU if args.compiler == "GNU" else Compiler.CRAY
#
#    for cfg_idx, configuration in enumerate(configurations):
#        assert configuration.job_time is not None
#        assert configuration.job_bin is not None
#
#        with open(f"{cfg_idx:04}.INP", "w") as file:
#            file.write(configuration.to_string())
#
#        boundary = configuration.boundary
#        assert boundary is not None
#
#        domain_x = boundary.domain_x
#        domain_y = boundary.domain_y
#        domain_z = boundary.domain_z
#        assert domain_x is not None
#        assert domain_y is not None
#        assert domain_z is not None
#
#        ndomains = domain_x * domain_y * domain_z
#        parallel = PARALLEL[ndomains]
#
#        jobs: list[Job] = []
#        for job_idx, (
#            nodes,
#            ntasks_per_node,
#            cpus_per_task,
#            gpus_per_node,
#        ) in enumerate(parallel):
#            jobs.append(
#                Job(
#                    cfg_idx,
#                    job_idx,
#                    nodes,
#                    ntasks_per_node,
#                    cpus_per_task,
#                    gpus_per_node,
#                    compiler,
#                    configuration.job_time,
#                    configuration.job_bin,
#                )
#            )
#
#        for job in jobs:
#            with open(f"{cfg_idx:04}-{job.job_idx:04}.LUMI.SLURM", "w") as file:
#                file.write(job.to_string())


if __name__ == "__main__":
    main()
