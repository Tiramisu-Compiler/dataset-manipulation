import sys, os, subprocess
from pathlib import Path
from datetime import datetime
import json
import time
from utils import load_task_info, write_task_info, datetime, read_stop_signal, launch_step, reset_stop_signal

tiramisu_path = '/data/scratch/mmerouani/tiramisu3/tiramisu'
os.environ['TIRAMISU_ROOT'] = tiramisu_path
os.environ['BEAM_SIZE'] = "3"  # auto-scheduler parameter
os.environ['MAX_DEPTH'] = "3"  # auto-scheduler parameter
os.environ['MAX_RUNS'] = "30"  # Maximum number of execution time measurments made on a schedule
os.environ['MIN_RUNS'] = "3"  # Minumum number of execution time measurments made on a schedule
os.environ['EXPLORE_BY_EXECUTION'] = "1"  # evalute the schedules by exploration
os.environ['SET_DEFAULT_EVALUATION'] = "0"  # evalute the schedules by exploration
os.environ['PRUNE_SLOW_SCHEDULES'] = "0"  # evalute the schedules by exploration
os.environ['SAVE_BEST_SCHED_IN_FILE'] = "0"  # save best speed up in file
os.environ[
    'EXECUTE_BEST_AND_INITIAL_SCHED'] = "1"  # execute the best schedule to get the real speed up, should only be used with the cost model
os.environ[
    'LOG_FILE_PATH'] = "/data/commit/tiramisu/data_factory_kb4083/in_progress_exec_large_beam_corrected_skew/benchmarks.txt"  # path where to save results
os.environ['AS_VERBOSE'] = "1"  # print outputs
# os.environ['EVAL_TIMEOUT'] = str(os.environ['NB_EXEC']*100) #seconds timeout for the no schedule version of the program
os.environ['INITIAL_TIMEOUT'] = "300000"  # seconds timeout for the no schedule version of the program
os.environ['SCHED_TIMEOUT_FACTOR'] = "100"  # The max slowdown factor that has the possibility to run MAX_RUNS times
os.environ['DYNAMIC_RUNS'] = "1"  # Activate the automatic adjustment of the number of runs

prepare_generator_cmd = 'cd ${FUNC_NAME};\
c++ -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/isl/include  -Wl,--no-as-needed -ldl -g -fno-rtti -lz -lpthread -std=c++11 -O0 -o ${FUNC_NAME}_generator.cpp.o -c ${FUNC_NAME}_generator.cpp;\
c++ -Wl,--no-as-needed -ldl -g -fno-rtti -lz -lpthread -std=c++11 -O0 ${FUNC_NAME}_generator.cpp.o -o ./${FUNC_NAME}_generator   -L${TIRAMISU_ROOT}/build  -L${TIRAMISU_ROOT}/3rdParty/Halide/lib  -L${TIRAMISU_ROOT}/3rdParty/isl/build/lib  -Wl,-rpath,${TIRAMISU_ROOT}/build:${TIRAMISU_ROOT}/3rdParty/Halide/lib:${TIRAMISU_ROOT}/3rdParty/isl/build/lib -ltiramisu -ltiramisu_auto_scheduler -lHalide -lisl'
run_generator_cmd = 'cd ${FUNC_NAME};\
./${FUNC_NAME}_generator'

prepare_autosched_cmd = 'cd ${FUNC_NAME};\
c++ -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/isl/include  -Wl,--no-as-needed -ldl -g -fno-rtti -lz -lpthread -std=c++11 -O0 -o ${FUNC_NAME}_autoscheduler.cpp.o -c ${FUNC_NAME}_autoscheduler.cpp;\
c++ -Wl,--no-as-needed -ldl -g -fno-rtti -lz -lpthread -std=c++11 -O0 ${FUNC_NAME}_autoscheduler.cpp.o -o ./${FUNC_NAME}_autoscheduler   -L${TIRAMISU_ROOT}/build  -L${TIRAMISU_ROOT}/3rdParty/Halide/lib  -L${TIRAMISU_ROOT}/3rdParty/isl/build/lib  -Wl,-rpath,${TIRAMISU_ROOT}/build:${TIRAMISU_ROOT}/3rdParty/Halide/lib:${TIRAMISU_ROOT}/3rdParty/isl/build/lib -ltiramisu -ltiramisu_auto_scheduler -lHalide -lisl ;\
g++ -shared -o ${FUNC_NAME}.o.so ${FUNC_NAME}.o;\
g++ -std=c++11 -fno-rtti -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/3rdParty/isl/include/ -I${TIRAMISU_ROOT}/benchmarks -L${TIRAMISU_ROOT}/build -L${TIRAMISU_ROOT}/3rdParty/Halide/lib/ -L${TIRAMISU_ROOT}/3rdParty/isl/build/lib -o ${FUNC_NAME}_wrapper -ltiramisu -lHalide -ldl -lpthread -lz -lm -Wl,-rpath,${TIRAMISU_ROOT}/build ./${FUNC_NAME}_wrapper.cpp ./${FUNC_NAME}.o.so -ltiramisu -lHalide -ldl -lpthread -lz -lm'
run_autosched_cmd = 'cd ${FUNC_NAME};\
./${FUNC_NAME}_autoscheduler > output.txt'
clean_up_cmd = 'cd ${FUNC_NAME};\
rm *.o *.o.*;'

if __name__ == "__main__":

    working_folder_path = sys.argv[1]
    working_folder_path = Path(working_folder_path)
    worker_id = int(working_folder_path.parts[-1][-2:])

    node_name = os.getenv('SLURMD_NODENAME')
    job_id = os.getenv('SLURM_JOBID')

    taskinfo = load_task_info(str(working_folder_path))
    taskinfo['is_active'] = True
    taskinfo['worker_name'] = node_name
    taskinfo['job_id'] = job_id
    taskinfo['last_updated'] = datetime.now()
    write_task_info(str(working_folder_path), taskinfo)

    print(
        f'\n# {str(datetime.now())} ---> Started autoscheduler script. Node:{node_name} Folder:{str(working_folder_path)}  \n',
        file=sys.stderr, flush=True)
    print(
        f'\n# {str(datetime.now())} ---> Started autoscheduler script. Node:{node_name} Folder:{str(working_folder_path)}  \n',
        file=sys.stdout, flush=True)

    while read_stop_signal(str(working_folder_path)) == False:
        func_folders = sorted(
            [e for e in working_folder_path.iterdir() if (e.is_dir() and str(e.parts[-1]).startswith('function'))],
            reverse=False)

        if len(func_folders) != 0:
            current_func_path = func_folders[0]
            taskinfo['current_function'] = str(current_func_path.parts[-1])
            function_name = taskinfo['current_function']
            taskinfo['last_updated'] = datetime.now()
            write_task_info(str(working_folder_path), taskinfo)
            os.environ['FUNC_NAME'] = function_name
            output_folder = Path(str(working_folder_path) + '/output')
            done_folder = Path(str(output_folder) + '/done')
            skipped_folder = Path(str(output_folder) + '/skipped')
            function_folder = Path(str(working_folder_path) + '/' + function_name)
            print(
                f'\n# {str(datetime.now())} ---> Working on {function_name}. Functions remaining {str(len(func_folders) - 1)} \n',
                flush=True)

            failed = launch_step('preparing generator', prepare_generator_cmd, function_name)

            if not failed:
                failed = launch_step('running generator', run_generator_cmd, function_name)

            if not failed:
                failed = launch_step('preparing autoscheduler', prepare_autosched_cmd, function_name)

            if not failed:
                failed = launch_step('running autoscheduler', run_autosched_cmd, function_name)

            if not failed:
                try:
                    out = subprocess.run(clean_up_cmd, check=True, shell=True, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE)
                except Exception as e:
                    print(f'\n# {str(datetime.now())} ---> Error cleaning up of {function_name} \n' + e.stderr.decode(
                        'UTF-8'), file=sys.stderr, flush=True)

            if failed:
                print(f'\n# {str(datetime.now())} function skipped due to error', flush=True)
                function_folder.rename(
                    skipped_folder / function_name)  # monving the function to the skipped function folder
            else:
                print(f'\n# {str(datetime.now())} function completed', flush=True)
                function_folder.rename(done_folder / function_name)  # monving the function to the done function folder

            taskinfo['last_updated'] = datetime.now()
            write_task_info(str(working_folder_path), taskinfo)

        else:  # no function to process
            taskinfo['current_function'] = None
            taskinfo['last_updated'] = datetime.now()
            print(f'\n# {str(datetime.now())} ---> No function to process, waiting for more functions', flush=True)
            write_task_info(str(working_folder_path), taskinfo)
            time.sleep(15)

    # when stop signal recieved
    taskinfo['is_active'] = False
    taskinfo['worker_name'] = None
    taskinfo['job_id'] = None
    taskinfo['current_function'] = None
    taskinfo['last_updated'] = datetime.now()
    write_task_info(str(working_folder_path), taskinfo)
    reset_stop_signal(str(working_folder_path))
    print(f'\n# {str(datetime.now())} ---> Stop signal recieved, job {job_id} quitting.', flush=True)

