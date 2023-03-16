import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from simple_slurm import Slurm
from tqdm.notebook import tqdm

if __name__ == "__main__":  # this isn't ran when this file is imported
    sys.path.append("RandomTiramisu")
    from RandomTiramisu import TiramisuMaker


def init_task_info(path):
    taskinfo = dict(
        is_active=False,
        current_function=None,
        job_id=None,
        worker_name=None,
        last_updated=None
    )
    write_task_info(path, taskinfo)


def reset_stop_signal(path):
    with open(path + '/stop_signal.txt', 'w') as f:
        f.write('0')


def send_stop_signal(path):
    with open(path + '/stop_signal.txt', 'w') as f:
        f.write('1')


def read_stop_signal(path):
    with open(path + '/stop_signal.txt', 'r') as f:
        if (int(f.read()) == 0):
            return False
        else:
            return True


# def folder_is_active(path): #checks if a worker is currently running on this folder
#     taskinfo = load_task_info(path)
#     if not taskinfo['is_active']:
#         return False
#     elif taskinfo['last_updated']==None:
#         return False
# #     elif (datetime.now() - taskinfo['last_updated']).total_seconds() > task_timeout:
# #         return False
#     else:
#         return True

def load_task_info(path):
    with open(path + '/task_info.json', 'r') as f:
        taskinfo = f.read()
    taskinfo = json.loads(taskinfo)
    if taskinfo['last_updated'] != None:
        taskinfo['last_updated'] = datetime.strptime(taskinfo['last_updated'], '%Y-%m-%d %H:%M:%S.%f')
    return taskinfo


def write_task_info(path, taskinfo):
    if taskinfo['last_updated'] != None:
        taskinfo['last_updated'] = str(taskinfo['last_updated'])
    taskinfo = json.dumps(taskinfo, indent=4)
    with open(path + '/task_info.json', 'w') as f:
        f.write(taskinfo)


def launch_worker_folder(path, worker_id):
    if not worker_is_dead(path):
        print(f'Cannot launch a new worker, a worker is already running in {path}', file=sys.stderr)
        return
    job = Slurm(
        job_name='data_gen_' + str(worker_id).zfill(2),
        nodes=1,
        exclusive='',
        # partition='compute',
        # exclude='lanka04,lanka21,lanka24',
        #         exclude = 'lanka04,lanka03,lanka21',
        #         open_mode = 'append',
        #         output= str(path)+ '/output_' + str(worker_id).zfill(2) + '/' + f'{Slurm.SLURM_JOBID}_{Slurm.SLURMD_NODENAME}_{Slurm.SLURM_JOB_NAME}.out',
        #         error = str(path)+ '/output_' + str(worker_id).zfill(2) + '/' + f'{Slurm.SLURM_JOBID}_{Slurm.SLURMD_NODENAME}_{Slurm.SLURM_JOB_NAME}.err',
        #         output= str(path)+ '/output_' + str(worker_id).zfill(2) + '/' + f'{Slurm.SLURM_JOB_NAME}.out',
        output=str(path) + '/output/data_gen_' + f'{str(worker_id).zfill(2)}.out',
        #         error = str(path)+ '/output_' + str(worker_id).zfill(2) + '/' + f'{Slurm.SLURM_JOB_NAME}.err'
        error=str(path) + '/output/data_gen_' + f'{str(worker_id).zfill(2)}.err'
    )

    job_id = job.sbatch(f'''
set -o allexport
source {data_factory_path}/.env
set +o allexport

. $CONDA_DIR/bin/activate
conda activate $CONDA_ENV


cd {path}

printenv CXX
printenv GXX


set -o allexport
source {data_factory_path}/.env
set +o allexport


printenv CXX
printenv GXX

stdbuf -o0 -e0 python3 {data_factory_path}/worker_script.py {path}
    ''')
        # 'cd ' + path + '; stdbuf -o0 -e0 python3 ' + str(data_factory_path + 'worker_script.py ') + str(path))
    print(f'Worker job {job_id} submitted on folder {str(path)}')


def launch_missing_workers():  # searches for folder with (unfinished functions and no worker working on them) and launches new workers
    worker_folders = sorted(
        [e for e in Path(in_progress_folder).iterdir() if (e.is_dir() and str(e.parts[-1]).startswith('worker'))])
    for worker_folder in worker_folders:
        worker_info = load_task_info(str(worker_folder))
        worker_id = int(str(worker_folder.parts[-1][-2:]))
        functions_left = len(
            [e for e in Path(worker_folder).iterdir() if (e.is_dir() and str(e.parts[-1]).startswith('function'))])
        if functions_left == 0:
            continue
        if not worker_info['is_active']:
            launch_worker_folder(str(worker_folder), worker_id)
            continue
        if worker_is_dead(str(worker_folder)):
            launch_worker_folder(str(worker_folder), worker_id)


def restart_active_workers(worker_ids='all'):  # relaunch the active workers after finishig their current function
    if worker_ids == 'all':
        worker_folders = sorted(
            [e for e in Path(in_progress_folder).iterdir() if (e.is_dir() and str(e.parts[-1]).startswith('worker'))])
    else:
        worker_folders = sorted([e for e in Path(in_progress_folder).iterdir() if (
                    e.is_dir() and str(e.parts[-1]).startswith('worker') and int(e.parts[-1][-2:]) in worker_ids)])
    active_workers = [worker_folder for worker_folder in worker_folders if not worker_is_dead(str(worker_folder))]
    stop_workers([int(worker_folder.parts[-1][-2:]) for worker_folder in active_workers])
    for worker_folder in active_workers:
        pid = os.fork()
        if pid == 0:
            try:
                print(f'Process {os.getpid()} waiting for {str(worker_folder.parts[-1])} to finish for restart')
                while not worker_is_dead(str(worker_folder)):
                    time.sleep(2)
                else:
                    launch_worker_folder(path=str(worker_folder), worker_id=int(worker_folder.parts[-1][-2:]))
                    print(f'Process {os.getpid()} restarted {str(worker_folder.parts[-1])} folder')
                os._exit(0)
            except Exception as e:
                print(f'Process {os.getpid()} of {str(worker_folder.parts[-1])} exited with exception {str(e)}',
                      file=sys.stderr)
                os._exit(0)


def worker_is_dead(path):  # check wheter the job id is in squeue
    taskinfo = load_task_info(path)
    if taskinfo['job_id'] == None:
        return True
    squeue_output = subprocess.run('squeue', check=True, shell=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE).stdout.decode('UTF-8')
    if str(taskinfo['job_id']) in squeue_output:
        return False
    else:
        return True


def stop_workers(worker_ids='all'):
    if worker_ids == 'all':
        worker_folders = sorted(
            [e for e in Path(in_progress_folder).iterdir() if (e.is_dir() and str(e.parts[-1]).startswith('worker'))])
    else:
        worker_folders = sorted([e for e in Path(in_progress_folder).iterdir() if (
                    e.is_dir() and str(e.parts[-1]).startswith('worker') and int(e.parts[-1][-2:]) in worker_ids)])
    for worker_folder in worker_folders:
        send_stop_signal(str(worker_folder))


def cancel_stop_signal(worker_ids='all'):
    if worker_ids == 'all':
        worker_folders = sorted(
            [e for e in Path(in_progress_folder).iterdir() if (e.is_dir() and str(e.parts[-1]).startswith('worker'))])
    else:
        worker_folders = sorted([e for e in Path(in_progress_folder).iterdir() if (
                    e.is_dir() and str(e.parts[-1]).startswith('worker') and int(e.parts[-1][-2:]) in worker_ids)])
    for worker_folder in worker_folders:
        reset_stop_signal(str(worker_folder))


def kill_workers(worker_ids='all'):
    workers_status = get_workers_status(count_dps=False)
    active_workers = workers_status[workers_status['is_running']]
    if worker_ids == 'all':
        if len(list(active_workers['job_id'])) == 0:
            print('No worker is currently active')
            return
        for job_id in list(active_workers['job_id']):
            subprocess.run('scancel ' + str(job_id), check=True, shell=True)
    else:
        for worker_id in worker_ids:
            if not worker_id in list(active_workers['worker_id']):
                print(f'Worker {worker_id} is already inactive')
                continue
            job_id = list(active_workers[active_workers['worker_id'] == worker_id]['job_id'])[0]
            subprocess.run('scancel ' + str(job_id), check=True, shell=True)


def get_workers_status(
        count_dps=True):  # prints the status of all worker folders, active, machine name, nb_prog remaining to exec, nb_prog in output, nb_errors,
    worker_folders = sorted(
        [e for e in Path(in_progress_folder).iterdir() if (e.is_dir() and str(e.parts[-1]).startswith('worker'))])
    worker_info_list = []
    total_number_of_dps = 0
    for worker_folder in tqdm(worker_folders):
        worker_info = load_task_info(str(worker_folder))
        worker_info['worker_id'] = int(str(worker_folder.parts[-1][-2:]))
        worker_info['functions_left'] = len(
            [e for e in Path(worker_folder).iterdir() if (e.is_dir() and str(e.parts[-1]).startswith('function'))])
        worker_info['functions_done'] = len([e for e in Path(worker_folder / 'output' / 'done').iterdir() if
                                             (e.is_dir() and str(e.parts[-1]).startswith('function'))])
        worker_info['functions_skipped'] = len([e for e in Path(worker_folder / 'output' / 'skipped').iterdir() if
                                                (e.is_dir() and str(e.parts[-1]).startswith('function'))])
        worker_info['stop_signal'] = read_stop_signal(str(worker_folder))
        if worker_info['last_updated'] != None:
            worker_info['time_since_update'] = str(
                timedelta(seconds=(datetime.now() - worker_info['last_updated']).total_seconds()))
        else:
            worker_info['time_since_update'] = None
        worker_info['is_running'] = not worker_is_dead(str(worker_folder))
        if count_dps:
            worker_info['dps_ready'] = get_nb_scheds_done(str(worker_folder))
            total_number_of_dps += worker_info['dps_ready']
        else:
            worker_info['dps_ready'] = '?'
        if (not worker_info['is_running']) and worker_info['is_active']:  # abrupt exit detected
            #             worker_info['is_active'] = False
            worker_info['job_id'] = None
            worker_info['worker_name'] = None
            worker_info['current_function'] = None
            print('Abrupt exit detected on worker ', worker_info['worker_id'])
        worker_info_list.append(worker_info)
    workers_status = pd.DataFrame(worker_info_list, columns=['worker_id', 'is_running', 'job_id', 'worker_name',
                                                             'current_function', 'time_since_update', 'functions_left',
                                                             'functions_done', 'functions_skipped', 'dps_ready',
                                                             'stop_signal'])
    if (count_dps): print("Total number of datapoints generated: ", total_number_of_dps)
    return workers_status


def get_nb_scheds_done(worker_path):
    total_scheds = 0
    for func_path in [e for e in Path(worker_path + '/output/done').iterdir() if
                      (e.is_dir() and str(e.parts[-1]).startswith('function'))]:
        func_name = str(func_path.parts[-1])
        if '_' in func_name:
            print(func_path)
        with open(str(func_path) + '/' + func_name + '_explored_schedules.json', 'r') as f:
            try:
                nb_scheds = len(json.loads(f.read())['schedules_list'])
            except:
                print("Couldn't read json file of", func_path)
                continue
        #             nb_scheds = len(json.loads(f.read())['schedules_list'])-1 # temprarily for rare scheds expr, -1 for not counting the no_sched
        #             nb_scheds = len(set(re.findall(r'\"id\": (\d+), \"schedule\": \"[\d\w,\(\)\-\s]*\",',f.read())))
        total_scheds += nb_scheds
    return total_scheds


def clear_skipped_funcs(mode='move', worker_ids='all'):
    if worker_ids == 'all':
        worker_folders = sorted(
            [e for e in Path(in_progress_folder).iterdir() if (e.is_dir() and str(e.parts[-1]).startswith('worker'))])
    else:
        worker_folders = sorted([e for e in Path(in_progress_folder).iterdir() if (
                    e.is_dir() and str(e.parts[-1]).startswith('worker') and int(e.parts[-1][-2:]) in worker_ids)])

    for worker_folder in worker_folders:
        if mode == 'move':
            for func_path in Path(worker_folder / 'output' / 'skipped').iterdir():
                func_path.rename(data_factory_path + '/processed/skipped/' + str(func_path.parts[-1]))
        elif mode == 'delete':
            shutil.rmtree(str(worker_folder) + '/output/skipped/')
            Path(str(worker_folder) + '/output/skipped/').mkdir()


def clear_done_funcs(mode='move', worker_ids='all'):
    if worker_ids == 'all':
        worker_folders = sorted(
            [e for e in Path(in_progress_folder).iterdir() if (e.is_dir() and str(e.parts[-1]).startswith('worker'))])
    else:
        worker_folders = sorted([e for e in Path(in_progress_folder).iterdir() if (
                    e.is_dir() and str(e.parts[-1]).startswith('worker') and int(e.parts[-1][-2:]) in worker_ids)])

    for worker_folder in worker_folders:
        if mode == 'move':
            for func_path in Path(worker_folder / 'output' / 'done').iterdir():
                func_path.rename(data_factory_path + '/processed/done/' + str(func_path.parts[-1]))
        elif mode == 'delete':
            shutil.rmtree(str(worker_folder) + '/output/done/')
            Path(str(worker_folder) + '/output/done/').mkdir()


# def delete_done_funcs(worker_ids='all'):
#     if worker_ids == 'all':
#         worker_folders = sorted([e for e in Path(in_progress_folder).iterdir() if (e.is_dir() and str(e.parts[-1]).startswith('worker'))])
#     else:
#         worker_folders = sorted([e for e in Path(in_progress_folder).iterdir() if (e.is_dir() and str(e.parts[-1]).startswith('worker') and int(e.parts[-1][-2:]) in worker_ids)])
#     for worker_folder in worker_folders:
#         shutil.rmtree(str(worker_folder)+'/output/done/')
#         Path(str(worker_folder)+'/output/done/').mkdir()


def print_error_file(worker_id, nb_lines=10):
    with open(in_progress_folder + '/worker_' + str(worker_id).zfill(2) + '/output/data_gen_' + str(worker_id).zfill(
            2) + '.err') as f:
        lines = f.readlines()
        print(f'Printing {nb_lines} last lines of worker {str(worker_id).zfill(2)} error file \n ------------------')
        print(''.join(lines[-nb_lines:]))


def print_log_file(worker_id, nb_lines=10):
    with open(in_progress_folder + '/worker_' + str(worker_id).zfill(2) + '/output/data_gen_' + str(worker_id).zfill(
            2) + '.out') as f:
        lines = f.readlines()
        print(f'Printing {nb_lines} last lines of worker {str(worker_id).zfill(2)} log file  \n-------------------')
        print(''.join(lines[-nb_lines:]))


# def merge_new_points(worker_ids='all', dataset_name='all_data.json'):
#     if worker_ids == 'all':
#         worker_folders = sorted([e for e in Path(in_progress_folder).iterdir() if (e.is_dir() and str(e.parts[-1]).startswith('worker'))])
#     else:
#         worker_folders = sorted([e for e in Path(in_progress_folder).iterdir() if (e.is_dir() and str(e.parts[-1]).startswith('worker') and int(e.parts[-1][-2:]) in worker_ids)])

#     with open(datasets_folder+'/'+dataset_name, 'a+') as f:
#         f.seek(0) # not sure if necessary
#         ds_content = f.read()
#         if len(ds_content) ==0: # if new dataset
#             ds_content='{}'
#         ds_dict = json.loads(ds_content)

#     for worker_folder in worker_folders:
#         for func_path in [e for e in Path(str(worker_folder)+'/output/done').iterdir() if (e.is_dir() and str(e.parts[-1]).startswith('function'))]:
#             func_name = str(func_path.parts[-1])

#             with open(str(func_path)+'/'+func_name+'_explored_schedules.json', 'r') as f:
#                 func_dict = json.loads(f.read())
#             ds_dict[func_name] = func_dict
#             # Format the explored_schedules.json to make it readable
#             formated_func_json = json.dumps(func_dict, indent = 4)
#             formated_func_json = re.sub(r'\[[\s*\w*,\"]*\]', format_array, formated_func_json)
#             with open(str(func_path)+'/'+func_name+'_explored_schedules.json', 'w') as f:
#                 f.write(formated_func_json)
#             # Move the function to the done folder
#             func_path.rename(data_factory_path+'/processed/done/'+str(func_path.parts[-1]))

#       # write dataset, PB, truncating the dataset before rewriting can be dangerous
#     with open(datasets_folder+'/'+dataset_name, 'w') as f:
#         f.write(json.dumps(ds_dict, indent = 4))

def format_array(array_str):
    array_str = array_str.group()
    array_str = array_str.replace('\n', '')
    array_str = array_str.replace('\t', '')
    array_str = array_str.replace('  ', '')
    array_str = array_str.replace(',', ', ')
    return array_str


def create_worker_folder(worker_id):  # creates and initializes a new worker folder
    worker_folder = Path(in_progress_folder + '/worker_' + str(worker_id).zfill(2))
    worker_folder.mkdir(parents=True, exist_ok=False)
    output_folder = Path(str(worker_folder) + '/output')
    output_folder.mkdir(parents=True, exist_ok=False)
    done_folder = Path(str(output_folder) + '/done')
    done_folder.mkdir(parents=True, exist_ok=False)
    skipped_folder = Path(str(output_folder) + '/skipped')
    skipped_folder.mkdir(parents=True, exist_ok=False)
    init_task_info(str(worker_folder))
    reset_stop_signal(str(worker_folder))
    return str(worker_folder)


def recycle_worker_folder(worker_id,
                          delete_folder=True):  # Move the processed to processed folder, remaining functions are sent to worker_0, delete folder is the param is set to true
    worker_folder = in_progress_folder + '/worker_' + str(worker_id).zfill(2)
    assert worker_is_dead(worker_folder)
    clear_skipped_funcs(mode='move', worker_ids=[worker_id])
    clear_done_funcs(mode='move', worker_ids=[worker_id])
    if worker_id != 0:
        for func_path in [e for e in Path(str(worker_folder)).iterdir() if
                          (e.is_dir() and str(e.parts[-1]).startswith('function'))]:
            func_name = str(func_path.parts[-1])
            func_path.rename(in_progress_folder + '/worker_00/' + str(func_path.parts[-1]))
        if delete_folder:
            shutil.rmtree(str(worker_folder))


def redistribute_functions(
        dest_worker_ids='all'):  # distribute remaining functions to existing worker folders with no stop signal set

    worker_folders = sorted(
        [e for e in Path(in_progress_folder).iterdir() if (e.is_dir() and str(e.parts[-1]).startswith('worker'))])
    if dest_worker_ids == 'all':
        dest_worker_ids = list(range(len(worker_folders)))
    #     print('dest_worker_ids',dest_worker_ids)
    #     stopped_worker_folders = filter(lambda x: read_stop_signal(x), worker_folders)
    destination_worker_folders = sorted([e for e in Path(in_progress_folder).iterdir() if (
                e.is_dir() and str(e.parts[-1]).startswith('worker') and int(e.parts[-1][-2:]) in dest_worker_ids)])
    #     destination_worker_folders = filter(lambda x: not read_stop_signal(x), worker_folders)
    folders_to_drain = list(filter(lambda x: not x in destination_worker_folders, worker_folders))
    assert len(destination_worker_folders) + len(folders_to_drain) == len(worker_folders)
    funcs_from_drain = []
    for worker_folder in folders_to_drain:
        ignore_func = ''
        if not worker_is_dead(str(worker_folder)):
            taskinfo = load_task_info(str(worker_folder))
            if taskinfo['current_function'] != None:
                ignore_func = taskinfo['current_function']
        funcs_from_drain.extend([e for e in worker_folder.iterdir() if (
                    e.is_dir() and str(e.parts[-1]).startswith('function') and str(e.parts[-1]) != ignore_func)])
    nb_funcs_dest = 0
    for worker_folder in destination_worker_folders:
        nb = get_nb_funcs_remaining(str(worker_folder))
        nb_funcs_dest += nb
    nb_funcs_per_worker = []
    for i in range(len(destination_worker_folders)):
        nb_funcs_per_worker.append(math.ceil(
            (nb_funcs_dest + len(funcs_from_drain) - sum(nb_funcs_per_worker)) / (len(destination_worker_folders) - i)))
    funcs_to_distribute = []
    funcs_to_distribute.extend(funcs_from_drain)
    for i, worker_folder in enumerate(destination_worker_folders):
        funcs_to_distribute.extend(
            sorted([e for e in worker_folder.iterdir() if (e.is_dir() and str(e.parts[-1]).startswith('function'))])[
            nb_funcs_per_worker[i]:])
    for i, worker_folder in enumerate(destination_worker_folders):
        nb_func_to_add = nb_funcs_per_worker[i] - get_nb_funcs_remaining(str(worker_folder))
        if nb_func_to_add > 0:
            funcs_to_add = funcs_to_distribute[:nb_func_to_add]
            for func in funcs_to_add:
                funcs_to_distribute.remove(func)
                func_name = str(func.parts[-1])
                func.rename(str(worker_folder) + '/' + func_name)


def generate_programs(nb_program):
    worker_folders = sorted(
        [e for e in Path(in_progress_folder).iterdir() if (e.is_dir() and str(e.parts[-1]).startswith('worker'))])
    if len(worker_folders) == 0:  # no worker folder found
        worker_folders = [Path(create_worker_folder(0))]
    nb_workers = len(worker_folders)
    progs_per_worker = math.ceil(nb_program / nb_workers)
    for worker_folder in worker_folders:
        TiramisuMaker.generate_programs(output_path=str(worker_folder), first_seed='auto', nb_programs=progs_per_worker)
    print(
        f'Generated {progs_per_worker * nb_workers} program across workers: {", ".join([str(e.parts[-1]) for e in worker_folders])}')


def scale_workers(desired_nb_workers):
    worker_folders = sorted(
        [e for e in Path(in_progress_folder).iterdir() if (e.is_dir() and str(e.parts[-1]).startswith('worker'))])
    current_worker_ids = [int(i.parts[-1][-2:]) for i in worker_folders]
    nb_workers = len(worker_folders)
    if nb_workers > desired_nb_workers:  # scaling down
        worker_to_recycle = worker_folders[-(nb_workers - desired_nb_workers):]
        print((nb_workers - desired_nb_workers))
        stop_workers(worker_ids=[int(e.parts[-1][-2:]) for e in worker_to_recycle])
        for worker_folder in worker_to_recycle:
            pid = os.fork()
            if pid == 0:
                try:
                    print(f'Process {os.getpid()} waiting for {str(worker_folder.parts[-1])} to finish')
                    while (get_nb_funcs_remaining(str(worker_folder)) > 0) and not worker_is_dead(str(worker_folder)):
                        time.sleep(1)
                    else:
                        if not worker_is_dead(str(worker_folder)):
                            time.sleep(15)  # make sure that the worker had time to processes the stop signal
                        recycle_worker_folder(worker_id=int(worker_folder.parts[-1][-2:]), delete_folder=True)
                        print(f'Process {os.getpid()} recycled {str(worker_folder.parts[-1])} folder')
                    os._exit(0)
                except Exception as e:
                    print(f'Process {os.getpid()} of {str(worker_folder.parts[-1])} exited with exception {str(e)}',
                          file=sys.stderr)
                    os._exit(0)

        time.sleep(3)  # just to make sure extra folders are recycled before redistributing functions
        redistribute_functions(dest_worker_ids=list(range(desired_nb_workers)))
    elif nb_workers < desired_nb_workers:  # scaling up
        missing_worker_ids = [i for i in range(desired_nb_workers) if not i in current_worker_ids]
        new_worker_folders = [create_worker_folder(worker_id) for worker_id in missing_worker_ids]
        redistribute_functions(dest_worker_ids='all')
    elif nb_workers == desired_nb_workers:  # just redistibute and launch missing workers
        redistribute_functions(dest_worker_ids='all')

    launch_missing_workers()


def get_nb_funcs_remaining(path):  # return the number of functions in progress
    return len([e for e in Path(str(path)).iterdir() if (e.is_dir() and str(e.parts[-1]).startswith('function'))])


def launch_step(step_str, step_cmd, function_name):
    failed = False
    try:
        out = subprocess.run(step_cmd, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        print(f'\n# {str(datetime.now())} ---> Error {step_str} of {function_name} \n' + e.stderr.decode('UTF-8'),
              file=sys.stderr, flush=True)
        out = e
        failed = True
    else:  # no exception rised
        if 'error' in out.stderr.decode('UTF-8'):
            print(f'\n# {str(datetime.now())} ---> Error {step_str} of {function_name} \n' + out.stderr.decode('UTF-8'),
                  file=sys.stderr, flush=True)
            failed = True
    if failed:
        with open(function_name + '/error.txt', 'a') as f:
            f.write('\nError while ' + step_str + '\n---------------------------\n' + out.stderr.decode('UTF-8') + '\n')
    return failed


def stop_waiting_workers():
    worker_folders = sorted(
        [e for e in Path(in_progress_folder).iterdir() if (e.is_dir() and str(e.parts[-1]).startswith('worker'))])
    for worker_folder in worker_folders:
        if not worker_is_dead(str(worker_folder)):
            if get_nb_funcs_remaining(str(worker_folder)) == 0:
                send_stop_signal(str(worker_folder))
                print(f'Stop signal sent to worker {worker_folder.parts[-1][-2:].zfill(2)}')


def collect_outputs(worker_ids='all'):  # moves the processed function to the processed folder
    clear_skipped_funcs(mode='move', worker_ids=worker_ids)
    clear_done_funcs(mode='move', worker_ids=worker_ids)


def make_dataset(funcs_range='all', funcs_path=None, dataset_name=None, copy_to=None,
                 file_type='json'):  # funcs_range is a tuple (first_id, last_id) inclusive

    nb_programs = 0
    nb_scheds = 0
    assert file_type == 'json' or file_type == 'pkl'
    if funcs_path == None:
        funcs_path = data_factory_path + '/processed/done/'
    if funcs_range == 'all':
        funcs_list = sorted(
            [e for e in Path(funcs_path).iterdir() if (e.is_dir() and str(e.parts[-1]).startswith('function'))])
    else:
        funcs_list = sorted([e for e in Path(funcs_path).iterdir() if (
                    e.is_dir() and str(e.parts[-1]).startswith('function') and funcs_range[0] <= int(
                e.parts[-1][-6:]) <= funcs_range[1])])
    if len(funcs_list) == 0:
        print(f'No function found in the defined range {funcs_range}')
        print('Aborted')
        return
    if dataset_name == None:
        funcs_range = (int(funcs_list[0].parts[-1][-6:]),
                       int(funcs_list[-1].parts[-1][-6:]))  # update the funcs range to the actual funcs
        dataset_name = f'dataset_batch{funcs_range[0]}-{funcs_range[1]}.' + file_type

    if Path(datasets_folder + '/' + dataset_name).exists():
        if input(f'Dataset {datasets_folder + "/" + dataset_name} already exists. Overwrite it?') != 'yes':
            print('Aborted')
            return
    ds_dict = dict()
    nb_programs = len(funcs_list)
    for func_path in tqdm(funcs_list):
        func_name = str(func_path.parts[-1])
        with open(str(func_path) + '/' + func_name + '_explored_schedules.json', 'r') as f:
            try:
                func_dict = json.loads(f.read())
            except:
                print(func_path, "couldn't read json file, skipping.")
                continue
        ds_dict[func_name] = func_dict
        nb_scheds += len(func_dict['schedules_list'])

    if file_type == 'json':
        with open(datasets_folder + '/' + dataset_name, 'w') as f:
            dataset_dumps = json.dumps(ds_dict, indent=4)
            formated_dataset_json = re.sub(r'\[[\s*\w*,\"]*\]', format_array, dataset_dumps)
            f.write(formated_dataset_json)
    elif file_type == 'pkl':
        with open(datasets_folder + '/' + dataset_name, 'wb') as f:
            pickle.dump(ds_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'Created dataset file: {datasets_folder + "/" + dataset_name}')
    if copy_to != None:
        shutil.copyfile(datasets_folder + '/' + dataset_name, copy_to + '/' + dataset_name)
        print(f'A copy created at {copy_to + "/" + dataset_name}')
    print('Number of programs :', nb_programs, ', number of datapoints :', nb_scheds)


def make_dataset_filter(funcs_range='all', funcs_path=None, dataset_name=None, copy_to=None, file_type='json',
                        names_to_drop=[]):  # funcs_range is a tuple (first_id, last_id) inclusive

    nb_programs = 0
    nb_scheds = 0
    assert file_type == 'json' or file_type == 'pkl'
    if funcs_path == None:
        funcs_path = data_factory_path + '/processed/done/'
    if funcs_range == 'all':
        funcs_list = sorted(
            [e for e in Path(funcs_path).iterdir() if (e.is_dir() and str(e.parts[-1]).startswith('function'))])
    else:
        funcs_list = sorted([e for e in Path(funcs_path).iterdir() if (
                    e.is_dir() and str(e.parts[-1]).startswith('function') and funcs_range[0] <= int(
                e.parts[-1][-6:]) <= funcs_range[1])])
    if len(funcs_list) == 0:
        print(f'No function found in the defined range {funcs_range}')
        print('Aborted')
        return
    if dataset_name == None:
        funcs_range = (int(funcs_list[0].parts[-1][-6:]),
                       int(funcs_list[-1].parts[-1][-6:]))  # update the funcs range to the actual funcs
        dataset_name = f'dataset_batch{funcs_range[0]}-{funcs_range[1]}.' + file_type

    if Path(datasets_folder + '/' + dataset_name).exists():
        if input(f'Dataset {datasets_folder + "/" + dataset_name} already exists. Overwrite it?') != 'yes':
            print('Aborted')
            return
    ds_dict = dict()
    nb_programs = len(funcs_list)
    for func_path in tqdm(funcs_list):
        func_name = str(func_path.parts[-1])
        if func_name in names_to_drop:
            continue
        with open(str(func_path) + '/' + func_name + '_explored_schedules.json', 'r') as f:
            func_dict = json.loads(f.read())
        ds_dict[func_name] = func_dict
        nb_scheds += len(func_dict['schedules_list'])

    if file_type == 'json':
        with open(datasets_folder + '/' + dataset_name, 'w') as f:
            dataset_dumps = json.dumps(ds_dict, indent=4)
            formated_dataset_json = re.sub(r'\[[\s*\w*,\"]*\]', format_array, dataset_dumps)
            f.write(formated_dataset_json)
    elif file_type == 'pkl':
        with open(datasets_folder + '/' + dataset_name, 'wb') as f:
            pickle.dump(ds_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'Created dataset file: {datasets_folder + "/" + dataset_name}')
    if copy_to != None:
        shutil.copyfile(datasets_folder + '/' + dataset_name, copy_to + '/' + dataset_name)
        print(f'A copy created at {copy_to + "/" + dataset_name}')
    print('Number of programs :', nb_programs, ', number of datapoints :', nb_scheds)


def merge_datasets(datasets_filenames, output_file):  # merges multiple dataset json files into a single one
    full_dataset_dict = dict()
    for dataset_filename in datasets_filenames:
        if dataset_filename.endswith('json'):
            with open(dataset_filename, 'r') as f:
                dataset_str = f.read()
            programs_dict = json.loads(dataset_str)
        elif dataset_filename.endswith('pkl'):
            with open(dataset_filename, 'rb') as f:
                programs_dict = pickle.load(f)
        full_dataset_dict = {**full_dataset_dict, **programs_dict}
    if output_file.endswith('json'):
        with open(output_file, 'w') as f:
            dataset_dumps = json.dumps(full_dataset_dict, indent=4)
            formated_dataset_json = re.sub(r'\[[\s*\w*,\"]*\]', format_array, dataset_dumps)
            f.write(formated_dataset_json)
    elif output_file.endswith('pkl'):
        with open(output_file, 'wb') as f:
            pickle.dump(full_dataset_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'Created dataset file: {output_file}')


def remove_inf_from_annot(func_folders):
    for func_path in [Path(e) for e in func_folders]:
        func_name = str(func_path.parts[-1])
        with open(str(func_path) + '/' + func_name + '_explored_schedules.json', 'r') as f:
            file_content = f.read()
        file_content = file_content.replace(' inf,', ' null,')
        with open(str(func_path) + '/' + func_name + '_explored_schedules.json', 'w') as f:
            f.write(file_content)
        print('Fixed ', func_path)