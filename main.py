from ase.io import read
import json
from pathlib import Path
import numpy as np
import os
from sys import argv

from wfl.configset import ConfigSet, OutputSpec
from wfl.calculators.castep import Castep
from wfl.autoparallelize import RemoteInfo
from wfl.calculators import generic
from expyre.resources import Resources
from wfl.autoparallelize import AutoparaInfo

BASE_WORKDIR = Path('/work/e89/e89/twarford/.dft_runs')

_, XYZ_PATH, SLICE = argv

def submit(config_set,
           output_spec,
           castep_params, 
           wfl_params):
    resources = Resources(
            max_time=wfl_params['max_time'],
            num_nodes=wfl_params['num_nodes'],
            partitions =wfl_params['partition'],
            )

    remote_info = RemoteInfo(
            ignore_failed_jobs=wfl_params['ignore_failed_jobs'],
            sys_name = "highmem" if wfl_params['partition'] == 'highmem' else "local", 
            job_name = 'castep', 
            resources = resources, 
            num_inputs_per_queued_job=1,
            input_files=['/work/e89/e89/twarford/castep_keywords.json'],
            pre_cmds=["module load castep/23.11"]) 
    
    compute = (Castep, [], castep_params)
    output_prefix = wfl_params['output_prefix']
    generic.calculate(
        inputs = config_set, 
        outputs = output_spec, 
        calculator = compute, 
        properties=["energy","forces","stress", 'magmoms'],
        output_prefix = f"{output_prefix}_", 
        autopara_info = AutoparaInfo(
            remote_info=remote_info,
            num_python_subprocesses = 1,
            num_inputs_per_python_subprocess=1,
        ),
        wait_for_results=wfl_params['wait_for_results'],
    )

def run_dft(xyz_path, slice):
    # io
    xyz_path = Path(xyz_path)
    print(f'Running DFT on {xyz_path}[{slice}]')
    configs = read(xyz_path, slice)
    castep_params_file = xyz_path.parent/'params_castep.json'
    wfl_params_file = xyz_path.parent/'params_wfl.json'
    if castep_params_file.exists():
        with open(castep_params_file, 'r') as castep_params_file:
            castep_params = json.load(castep_params_file)
    else:
        print(f'No CASTEP parameters file found in {xyz_path.parent}. Generating file.')
        castep_params = get_default_castep_params(configs)
    if wfl_params_file.exists():
        with open(wfl_params_file, 'r') as wfl_params_file:
            wfl_params = json.load(wfl_params_file)
    else:
        print(f'No workflow parameters file found in {xyz_path.parent}. Generating file.')
        wfl_params = get_default_wfl_params(configs, castep_params)

    output_dir = xyz_path.parent/wfl_params['output_prefix']; output_dir.mkdir(exist_ok=True)
    output_spec = OutputSpec(output_dir/f'output[{slice}].xyz', overwrite=wfl_params['overwrite'])
    config_set = ConfigSet(configs)

    dft_dir = Path(castep_params['workdir']); dft_dir.mkdir(exist_ok=True)
    symlink_location = output_dir/'dft'
    if symlink_location.exists(): os.remove(symlink_location)    
    os.symlink(dft_dir, symlink_location, target_is_directory=True)
    # TODO: don't submit if too many jobs running 
    with open(output_dir/'params_castep.json', 'w') as f:
        json.dump(castep_params, f, indent=4)
    with open(output_dir/'params_wfl.json', 'w') as f:
        json.dump(wfl_params, f, indent=4)
    submit(config_set, output_spec, castep_params, wfl_params)

def get_default_castep_params(configs):
    if isinstance(configs, list):
        config = configs[0]
    else:
        config = configs

    castep_params = {}
    castep_params['task'] = 'singlepoint'
    castep_params['xc_functional'] = 'PBE'
    castep_params['kpoint_mp_grid'] = get_kpoints(config, 0.045)
    castep_params['perc_extra_bands'] = 50 # for a metal! 
    castep_params['spin_polarized'] = any(atom.magmom != 0 for atom in config)
    castep_params['cut_off_energy'] = 520
    castep_params['elec_method'] = 'edft'
    castep_params['smearing_scheme'] = 'gaussian'
    castep_params['smearing_width'] = 0.17
    castep_params['max_scf_cycles'] = 300
    castep_params["castep_command"] = "srun --distribution=block:block --hint=nomultithread castep.mpi"
    castep_params['rundir_prefix'] = f"{config.symbols}_{get_output_prefix(castep_params)}"
    castep_params['workdir'] = str(BASE_WORKDIR/castep_params['rundir_prefix'])
    return castep_params


def get_kpoints(atoms, spacing):
    """
    Generate a sensible k-point grid based on the unit cell lattice vectors.
    
    Parameters:
    - atoms: (3,3) array-like, real-space lattice vectors in Angstroms.
    - spacing: float, in Å^-1.
    
    Returns:
    - (3,) tuple of k-point grid sizes.
    """
    min_density = 1/spacing
    recip_vectors = atoms.cell.reciprocal()
    
    # Compute reciprocal lattice vector magnitudes
    magnitudes = np.linalg.norm(recip_vectors, axis=1)
    # Set grid size such that k-point spacing is approximately spacing
    k_grid = np.maximum(np.round(magnitudes * min_density).astype(int), 1)
    return tuple(int(n) for n in k_grid)

def get_default_wfl_params(configs, castep_params):
    if isinstance(configs, list):
        config = configs[0]
    else:
        config = configs
    num_atoms = len(config)
    wfl_params = {}
    wfl_params['output_prefix'] = get_output_prefix(castep_params)
    wfl_params['wait_for_results'] = True
    wfl_params['overwrite'] = False
    wfl_params['ignore_failed_jobs'] = True
    wfl_params['max_time'] = "20m" if (num_atoms < 10) and (castep_params['task']=='singlepoint') else "24h"
    wfl_params['partition'] = "short" if wfl_params['max_time'] == "20m" else "standard"
    num_kpoints = 1
    for n in castep_params['kpoint_mp_grid']:
        num_kpoints *= n
    wfl_params['num_nodes'] = max( min(num_kpoints // 128, num_atoms), 1)
    return wfl_params

def get_output_prefix(castep_params):
    output_prefix = castep_params['xc_functional']
    if "hubbard_u" in castep_params: output_prefix += 'PU'
    return output_prefix

if __name__ == '__main__':
    run_dft(XYZ_PATH, SLICE)


# def submit_castep_calculation(work_dir, config_set, castep_params_file, xc_functional, wait_for_results, overwrite):
#     castep_params = castep_params_file.copy()
#     if xc_functional == 'GGA': # Remove U from GGA calculations (if present)
#         castep_params.pop('hubbard_u', None)

#     output_spec = OutputSpec(work_dir/f'{xc_functional}.xyz', overwrite=overwrite)
#     num_kpoints = 1
#     for n in castep_params['kpoint_mp_grid']:
#         num_kpoints *= n
#     num_atoms = len(config_set.items[0]) 
#     num_nodes = min(math.ceil(num_kpoints / 128), num_atoms)
#     num_inputs_per_queued_job = 1
#     max_time = ("20m" if ((num_nodes > 1) or (num_atoms < 10)) and castep_params['task']=='singlepoint' else "24h") # Some relaxations (V) fail with 20m
#     partition = "short" if max_time == "20m" else "standard"
#     if 'Pd/slabs' in str(work_dir):
#         partition = "highmem"
#         num_nodes = 2 # Pd seems to use more than the predicted 3.2GB per process on 1 node!
#     print(f'{max_time=}', f'{partition=}')

#     resources = Resources(
#         max_time = max_time,
#         num_nodes= num_nodes,
#         partitions = partition,
#         )

#     remote_info = RemoteInfo(
#         # ignore_failed_jobs=True,
#         sys_name = "highmem" if partition == 'highmem' else "local", 
#         job_name = 'castep', 
#         resources = resources, 
#         num_inputs_per_queued_job=num_inputs_per_queued_job,
#         input_files=['/work/e89/e89/twarford/castep_keywords.json'],
#         pre_cmds=[
#         "module load craype-x86-genoa",
#         "module load cray-fftw/3.3.10.3",
#         "source /work/e89/e89/jj569/Venv/bin/activate",
#         "module load load-epcc-module",
#         "module load epcc-setup-env",
#         "module load castep/23.11"]) 

#     compute = (Castep, [], castep_params)

#     generic.calculate(
#         inputs = config_set, 
#         outputs = output_spec, 
#         calculator = compute, 
#         properties=["energy","forces","stress"],
#         output_prefix = f"{xc_functional}_", 
#         autopara_info = AutoparaInfo(
#             remote_info=remote_info,
#             num_python_subprocesses = 1,
#             num_inputs_per_python_subprocess=1,
#         ),
#         wait_for_results=wait_for_results,
#     )