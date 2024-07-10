import re
import matplotlib.pyplot as plt
import argparse

def extract_data_from_text(file_path):
    data = {'concurrency': [], 'throughput': [], 'latency': []}
    
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Define the regex patterns
    concurrency_pattern = re.compile(r'Request concurrency: (\d+)')
    throughput_pattern = re.compile(r'Throughput: ([\d.]+) infer/sec')
    latency_pattern = re.compile(r'Avg latency: (\d+) usec')
    
    # Find all matches
    concurrencies = concurrency_pattern.findall(content)
    throughputs = throughput_pattern.findall(content)
    latencies = latency_pattern.findall(content)
    
    # Convert string matches to appropriate types and store in the dictionary
    for concurrency, throughput, latency in zip(concurrencies, throughputs, latencies):
        data['concurrency'].append(int(concurrency))
        data['throughput'].append(float(throughput))
        data['latency'].append(int(latency))
    
    return data

def plot_data(cpu_data, gpu_data, 
              variable='throughput',
              ylabel='Throughput (infer/sec)',
              save_path='plots',
              save_name='concurrency_vs_throughput.png'):
    
    plt.figure(figsize=(5, 5))
    plt.plot(cpu_data['concurrency'], cpu_data[variable], 'o-', label='CPU')
    plt.plot(gpu_data['concurrency'], gpu_data[variable], 's-', label='GPU')
    plt.xlabel('Concurrency', loc='right')
    plt.ylabel(ylabel, loc='top')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f'{save_path}/{save_name}.pdf')

def main():
    plot_data(cpu_data, gpu_data)
    plot_data(cpu_data, gpu_data, 
              variable='latency', 
              ylabel='Latency (usec)', 
              save_name='concurrency_vs_latency.png')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-igpu", "--input-gpu", 
                        default='data/gpu_concurrency_vs_throughput.log',
                        type=str, help="Input text file path")
    parser.add_argument("-icpu", "--input-cpu",
                        default='data/cpu_concurrency_vs_throughput.log',
                        type=str, help="Input text file path")
    args = parser.parse_args()
    
    cpu_data = extract_data_from_text(args.input_cpu)
    gpu_data = extract_data_from_text(args.input_gpu)
    
    main()