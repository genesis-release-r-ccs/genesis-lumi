import argparse
import copy
import dataclasses
import datetime
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
    AUTOSELECT = "AUTOSELECT"
    OPT_1DALLTOALL = "OPT_1DALLTOALL"
    OPT_2DALLTOALL = "OPT_2DALLTOALL"
    NOOPT_1DALLTOALL = "NOOPT_1DALLTOALL"
    NOOPT_2DALLTOALL = "NOOPT_2DALLTOALL"


class NonbondKernel(enum.Enum):
    AUTOSELECT = "AUTOSELECT"
    FUGAKU = "FUGAKU"
    INTEL = "INTEL"
    GENERIC = "GENERIC"
    GPU = "GPU"


class Integrator(enum.Enum):
    VVER = "VVER"
    VRES = "VRES"


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
    stoptr_period: typing.Optional[int] = None
    nbupdate_period: typing.Optional[int] = None
    elec_long_period: typing.Optional[int] = None
    thermostat_period: typing.Optional[int] = None
    barostat_period: typing.Optional[int] = None
    crdout_period: typing.Optional[int] = None
    rstout_period: typing.Optional[int] = None
    iseed: typing.Optional[int] = None


@dataclasses.dataclass
class ConstraintsBlock(Block):
    rigid_bond: typing.Optional[YesNo] = None
    water_model: typing.Optional[str] = None
    hydrogen_type: typing.Optional[str] = None
    fast_water: typing.Optional[YesNo] = None
    hydrogen_mass_upper_bound: typing.Optional[float] = None


@dataclasses.dataclass
class EnsembleBlock(Block):
    ensemble: typing.Optional[Ensemble] = None
    tpcontrol: typing.Optional[TPControl] = None
    pressure: typing.Optional[float] = None
    temperature: typing.Optional[float] = None
    group_tp: typing.Optional[YesNo] = None
    gamma_t: typing.Optional[float] = None
    tau_p: typing.Optional[float] = None


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
    cfg_name: typing.Optional[str] = None

    def to_string(self):
        string = ""
        for field in dataclasses.fields(self):
            item = getattr(self, field.name)
            if isinstance(item, Block):
                string += f"{item.to_string()}\n"
        return string


@dataclasses.dataclass
class Job:
    num_jobs: int = 1
    node_cpus: typing.Optional[int] = None
    compiler: typing.Optional[Compiler] = None
    time: typing.Optional[datetime.time] = None
    bin: typing.Optional[str] = None
    cfg_name: typing.Optional[str] = None
    job_name: typing.Optional[str] = None
    nodes: typing.Optional[int] = None
    mpis: typing.Optional[int] = None
    omps: typing.Optional[int] = None
    gpus: typing.Optional[int] = None

    def to_string(self):
        assert self.cfg_name is not None
        assert self.job_name is not None
        assert self.nodes is not None
        assert self.mpis is not None
        assert self.omps is not None
        assert self.gpus is not None
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
        string += f"#SBATCH --ntasks-per-node={self.mpis}\n"
        string += f"#SBATCH --cpus-per-task={self.omps}\n"
        string += f"#SBATCH --gpus-per-node={self.gpus}\n"
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
        match self.mpis:
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
        string += f"srun {self.bin} {self.cfg_name}.INP\n"
        string += "\n"
        string += "# Benchmark\n"
        for i in range(self.num_jobs):
            prefix = f"{self.job_name}-{i:04}"
            stdout = f"{prefix}.OUT"
            stderr = f"{prefix}.ERR"

            string += f"srun {self.bin} {self.cfg_name}.INP"
            string += f" 1> {stdout}"
            string += f" 2> {stderr}"
            string += "\n"

        return string


def build_configurations(toml: dict[str, typing.Any]) -> list[Configuration]:
    # Stage 1
    configuration = Configuration()

    # Stage 2
    input = toml["input"]
    input = {k: v for k, v in input.items() if not isinstance(v, list)}
    configuration.input = InputBlock(**input)

    energy = toml["energy"]
    energy = {k: v for k, v in energy.items() if not isinstance(v, list)}
    configuration.energy = EnergyBlock(**energy)

    dynamics = toml["dynamics"]
    dynamics = {k: v for k, v in dynamics.items() if not isinstance(v, list)}
    configuration.dynamics = DynamicsBlock(**dynamics)

    constraints = toml["constraints"]
    constraints = {k: v for k, v in constraints.items() if not isinstance(v, list)}
    configuration.constraints = ConstraintsBlock(**constraints)

    ensemble = toml["ensemble"]
    ensemble = {k: v for k, v in ensemble.items() if not isinstance(v, list)}
    configuration.ensemble = EnsembleBlock(**ensemble)

    boundary = toml["boundary"]
    boundary = {k: v for k, v in boundary.items() if not isinstance(v, list)}
    configuration.boundary = BoundaryBlock(**boundary)

    # Stage 3
    if "pme_scheme" in toml["energy"]:
        new_configurations: list[Configuration] = []
        for pme_scheme in toml["energy"]["pme_scheme"]:
            new_configuration = copy.deepcopy(configuration)
            new_configuration.energy.pme_scheme = PMEScheme(pme_scheme)
            new_configurations.append(new_configuration)
    else:
        new_configurations = [copy.deepcopy(configuration)]

    configurations = copy.deepcopy(new_configurations)

    # Stage 4
    if (
        "pme_ngrid_x" in toml["energy"]
        and "pme_ngrid_y" in toml["energy"]
        and "pme_ngrid_z" in toml["energy"]
        and "pme_nspline" in toml["energy"]
    ):
        new_configurations: list[Configuration] = []
        for pme_ngrid_x, pme_ngrid_y, pme_ngrid_z, pme_nspline in zip(
            toml["energy"]["pme_ngrid_x"],
            toml["energy"]["pme_ngrid_y"],
            toml["energy"]["pme_ngrid_z"],
            toml["energy"]["pme_nspline"],
        ):
            for configuration in configurations:
                new_configuration = copy.deepcopy(configuration)
                new_configuration.energy.pme_ngrid_x = pme_ngrid_x
                new_configuration.energy.pme_ngrid_y = pme_ngrid_y
                new_configuration.energy.pme_ngrid_z = pme_ngrid_z
                new_configuration.energy.pme_nspline = pme_nspline
                new_configurations.append(new_configuration)
    else:
        new_configurations = copy.deepcopy(configurations)

    configurations = copy.deepcopy(new_configurations)

    # Stage 5
    if "nonbond_kernel" in toml["energy"]:
        new_configurations: list[Configuration] = []
        for nonbond_kernel in toml["energy"]["nonbond_kernel"]:
            for configuration in configurations:
                new_configuration = copy.deepcopy(configuration)
                new_configuration.energy.nonbond_kernel = NonbondKernel(nonbond_kernel)
                new_configurations.append(new_configuration)
    else:
        new_configurations = copy.deepcopy(configurations)

    configurations = copy.deepcopy(new_configurations)

    # Stage 7
    new_configurations: list[Configuration] = []
    for domain_x in toml["boundary"]["domain_x"]:
        for domain_y in toml["boundary"]["domain_y"]:
            for domain_z in toml["boundary"]["domain_z"]:
                total_domains = domain_x * domain_y * domain_z
                if total_domains % 2 != 0:
                    continue
                if total_domains < 4:
                    continue
                for configuration in configurations:
                    new_configuration = copy.deepcopy(configuration)
                    new_configuration.boundary.domain_x = domain_x
                    new_configuration.boundary.domain_y = domain_y
                    new_configuration.boundary.domain_z = domain_z
                    new_configurations.append(new_configuration)

    configurations = copy.deepcopy(new_configurations)

    # Stage 8
    new_configurations: list[Configuration] = []
    for (
        integrator,
        nsteps,
        timestep,
        elec_long_period,
        thermostat_period,
        barostat_period,
        nbupdate_period,
    ) in zip(
        toml["dynamics"]["integrator"],
        toml["dynamics"]["nsteps"],
        toml["dynamics"]["timestep"],
        toml["dynamics"]["elec_long_period"],
        toml["dynamics"]["thermostat_period"],
        toml["dynamics"]["barostat_period"],
        toml["dynamics"]["nbupdate_period"],
    ):
        for configuration in configurations:
            new_configuration = copy.deepcopy(configuration)
            new_configuration.dynamics.integrator = Integrator(integrator)
            new_configuration.dynamics.nsteps = nsteps
            new_configuration.dynamics.timestep = timestep
            new_configuration.dynamics.elec_long_period = elec_long_period
            new_configuration.dynamics.thermostat_period = thermostat_period
            new_configuration.dynamics.barostat_period = barostat_period
            new_configuration.dynamics.nbupdate_period = nbupdate_period
            new_configurations.append(new_configuration)

    configurations = copy.deepcopy(new_configurations)

    # Add names to the configurations
    for configuration_id, configuration in enumerate(configurations):
        configuration.cfg_name = f"{configuration_id:04}"

    # Return the final configurations
    return configurations


def build_jobs(
    toml: dict[str, typing.Any], configurations: list[Configuration]
) -> list[Job]:
    job = toml["job"]
    job = {k: v for k, v in job.items() if not isinstance(v, list)}
    new_job = Job(**job)
    job_stage1 = copy.deepcopy(new_job)

    new_jobs: list[Job] = []
    for configuration in configurations:
        assert configuration.energy.nonbond_kernel is not None
        assert configuration.boundary.domain_x is not None
        assert configuration.boundary.domain_y is not None
        assert configuration.boundary.domain_z is not None
        domain_x = configuration.boundary.domain_x
        domain_y = configuration.boundary.domain_y
        domain_z = configuration.boundary.domain_z
        total_domains = domain_x * domain_y * domain_z
        total_cpus = job_stage1.node_cpus
        if configuration.energy.nonbond_kernel == NonbondKernel.GPU:
            use_gpu = True
        else:
            use_gpu = False

        nodes = toml["job"]["nodes"]
        mpis = toml["job"]["mpis"]
        omps = toml["job"]["omps"]

        job_counter = 0
        for num_nodes in nodes:
            for num_mpis in mpis:
                if num_nodes * num_mpis != total_domains:
                    continue
                for num_omps in omps:
                    if num_mpis * num_omps != total_cpus:
                        continue
                    new_job = copy.deepcopy(job_stage1)
                    new_job.cfg_name = configuration.cfg_name
                    new_job.nodes = num_nodes
                    new_job.mpis = num_mpis
                    new_job.omps = num_omps
                    new_job.gpus = num_mpis
                    new_job.job_name = f"{configuration.cfg_name}-{job_counter:04}"
                    assert new_job.bin is not None
                    new_job.bin += "_gpu" if use_gpu else "_cpu"
                    new_jobs.append(new_job)
                    job_counter += 1

    jobs_stage2 = copy.deepcopy(new_jobs)

    # Return the final jobs
    jobs = jobs_stage2
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT", type=str, help="Input file in TOML format")
    args = parser.parse_args()

    # Open toml file
    with open(args.INPUT, "rb") as file:
        toml = tomllib.load(file)

    # Build configurations
    configurations = build_configurations(toml)

    # Build jobs
    jobs = build_jobs(toml, configurations)

    # Write configurations
    for configuration in configurations:
        with open(f"{configuration.cfg_name}.INP", "w") as file:
            file.write(configuration.to_string())

    # Write jobs
    for job in jobs:
        with open(f"{job.job_name}.SLURM", "w") as file:
            file.write(job.to_string())


if __name__ == "__main__":
    main()
