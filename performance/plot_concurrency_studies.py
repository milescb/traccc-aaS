import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_data(cpu_data, gpu_data, 
              variable='Inferences/Second',
              ylabel='Throughput (infer/sec)',
              save_path='plots',
              save_name='concurrency_vs_throughput.pdf'):
    
    plt.figure(figsize=(5, 5))
    plt.plot(cpu_data['Concurrency'].values, cpu_data[variable].values, 'o-', label='CPU')
    plt.plot(gpu_data['Concurrency'].values, gpu_data[variable].values, 's-', label='GPU')
    plt.xlabel('Concurrency', loc='right')
    plt.ylabel(ylabel, loc='top')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f'{save_path}/{save_name}', bbox_inches='tight')
    
def gpu_to_number(df):
    for column in df.columns:
        if 'GPU' in column:
            df[column] = pd.to_numeric(df[column].str.split(':').str[-1].str.rstrip(';'), errors='coerce')
    return df

def main():
    plot_data(cpu_data, gpu_data)
    plot_data(cpu_data, gpu_data, 
              variable='Avg latency', 
              ylabel='Latency (usec)', 
              save_name='concurrency_vs_latency.pdf')
    plot_data(cpu_data, gpu_data, 
              variable='Avg GPU Utilization', 
              ylabel='Average GPU Utilization (%)',
              save_name='concurrency_vs_gpu_utilization.pdf')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-gpu", "--input-gpu", 
                        default='data/out_gpu.csv',
                        type=str, help="Input text file path")
    parser.add_argument("-cpu", "--input-cpu",
                        default='data/out_cpu.csv',
                        type=str, help="Input text file path")
    args = parser.parse_args()
    
    cpu_data = pd.read_csv(args.input_cpu)
    gpu_data = pd.read_csv(args.input_gpu)
    
    cpu_data = cpu_data.sort_values(by='Concurrency', ascending=True)
    gpu_data = gpu_data.sort_values(by='Concurrency', ascending=True)
    
    cpu_data = gpu_to_number(cpu_data)
    gpu_data = gpu_to_number(gpu_data)
    
    main()