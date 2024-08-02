import subprocess
import time
import os
import argparse

def run_command_in_tmux(session_name, window_name, command):
    # Check if the window exists
    check_command = f"tmux list-windows -t {session_name} | grep {window_name}"
    window_exists = subprocess.run(check_command, shell=True, stdout=subprocess.PIPE).returncode == 0
    
    # If the window does not exist, create it
    if not window_exists:
        subprocess.run(['tmux', 'new-window', '-t', session_name, '-n', window_name])
    
    # Send the command to the tmux window
    subprocess.run(['tmux', 'send-keys', '-t', f"{session_name}:{window_name}", command, 'C-m'])
    
def run_command_in_tmux_and_wait(session_name, window_name, command, flag_file="command_done.flag"):
    # Ensure the flag file does not exist before starting
    if os.path.exists(flag_file):
        os.remove(flag_file)
    
    # Modify the command to touch the flag file upon completion
    command += f" && touch {flag_file}"
    
    # Send the command to the tmux window
    subprocess.run(['tmux', 'send-keys', '-t', f"{session_name}:{window_name}", command, 'C-m'])
    
    # Wait for the flag file to appear
    while not os.path.exists(flag_file):
        time.sleep(1)  # Check every second

    # Optionally, remove the flag file after completion
    os.remove(flag_file)

def update_config_file(file_path, instance_count):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    with open(file_path, 'w') as file:
        for line in lines:
            if line.strip().startswith('count:'):
                file.write(f'    count: {instance_count}\n')
            else:
                file.write(line)

def run_perf_analyzer(model, input_data, concurrency_range, measurement_interval, output_file):
    command = (
        f"perf_analyzer -m {model} --input-data {input_data} "
        f"--concurrency-range {concurrency_range} --verbose-csv --collect-metrics "
        f"--measurement-interval {measurement_interval} -f {output_file}"
    )
    subprocess.run(command, shell=True)

def main():
    
    # Create a new tmux session
    session_name = "triton_perf"
    subprocess.run(['tmux', 'new-session', '-d', '-s', session_name])

    # Start Docker container in the tmux session
    docker_command = "shifter --module=gpu --image=docexoty/tritonserver:latest"
    source_command = "source ../setup.out"
    subprocess.run(source_command, shell=True)
    run_command_in_tmux(session_name, f"client", f"{docker_command}")
    time.sleep(10)
    run_command_in_tmux(session_name, f"client", f"{source_command}")
    time.sleep(5)
    
    run_command_in_tmux(session_name, f"server", f"{docker_command}")
    time.sleep(10)
    run_command_in_tmux(session_name, f"server", f"{source_command}")
    time.sleep(5)

    for instance_count in range(1, args.max_instances + 1):
        print(f"Running with instance count: {instance_count}")

        # Update the config file
        cpu_config_file = f"{args.model_repository}/traccc-cpu/config.pbtxt"
        gpu_config_file = f"{args.model_repository}/traccc-gpu/config.pbtxt"
        update_config_file(cpu_config_file, instance_count)
        update_config_file(gpu_config_file, instance_count)

        # Start Triton server
        triton_command = f"tritonserver --model-repository={args.model_repository}"
        run_command_in_tmux(session_name, f"server", f"{triton_command}")
        time.sleep(90)

        # Run perf_analyzer for GPU
        gpu_output_file = f"{args.output_directory}/gpu_{instance_count}instance.csv"
        run_command_in_tmux_and_wait(session_name, f"client", 
                            f"perf_analyzer -m traccc-gpu --input-data {args.input_data} "
                            f"--concurrency-range 1:{args.max_concurrency}:1 --verbose-csv --collect-metrics "
                            f"--measurement-interval {args.measurement_interval} -f {gpu_output_file}")

        # Run perf_analyzer for CPU
        cpu_output_file = f"{args.output_directory}/cpu_{instance_count}instance.csv"
        run_command_in_tmux_and_wait(session_name, f"client", 
                            f"perf_analyzer -m traccc-cpu --input-data {args.input_data} "
                            f"--concurrency-range 1:{args.max_concurrency}:1 --verbose-csv --collect-metrics "
                            f"--measurement-interval {args.measurement_interval} -f {cpu_output_file}")

        # Stop Triton server
        subprocess.run(f'tmux send-keys -t {session_name}:server C-c', shell=True)
        time.sleep(20)
    
    plot_command = f"python plot_model_instance_studies.py -i {args.output_directory} -o {args.output_directory}"   
    run_command_in_tmux_and_wait(session_name, f"client", plot_command)

    # Kill running processes
    subprocess.run(f'tmux send-keys -t {session_name}:client C-c', shell=True)
    time.sleep(2)
    subprocess.run(f'tmux send-keys -t {session_name}:client exit', shell=True)
    time.sleep(2)
    subprocess.run(f'tmux send-keys -t {session_name}:server C-c', shell=True)
    time.sleep(2)
    subprocess.run(f'tmux send-keys -t {session_name}:server exit', shell=True)
    time.sleep(2)
    
    # Kill the tmux session
    subprocess.run(['tmux', 'kill-session', '-t', session_name])

    print("All tests completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-data', type=str, required=True)
    parser.add_argument('-o', '--output-directory', type=str, required=True)
    parser.add_argument('--model-repository', type=str, default='../backend/models/')
    parser.add_argument('--measurement-interval', type=int, default=10000)
    parser.add_argument('--max-instances', type=int, default=8)
    parser.add_argument('--max-concurrency', type=int, default=5)
    args = parser.parse_args()
    
    main()
