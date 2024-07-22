import os, re
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def clean_pandas_df(df):
    df = df.sort_values(by='Concurrency', ascending=True)
    
    new_columns = {}
        
    for index, row in df.iterrows():
        # Split the cell content by ';' to get individual GPU utilizations
        gpus = row['Avg GPU Utilization'].rstrip(';').split(';')
        
        for i, gpu in enumerate(gpus):
            _, utilization = gpu.split(':')
            new_col_name = f'gpu_util_{i}'
            
            # Add the utilization value to the new_columns dictionary
            if new_col_name not in new_columns:
                new_columns[new_col_name] = [None] * len(df) 
            new_columns[new_col_name][index] = pd.to_numeric(utilization, errors='coerce') * 100
    
    # Add the new columns to the DataFrame
    for new_col_name, values in new_columns.items():
        df[new_col_name] = values
        
    gpu_util_columns = [col for col in df.columns if 'gpu_util_' in col]
    df['total_gpu_usage'] = df[gpu_util_columns].sum(axis=1)
        
    df.drop('Avg GPU Utilization', axis=1, inplace=True)
    
    return df

def instance_number(filename):
    match = re.search(r'(cpu|gpu)_(\d+)instance\.csv', filename)
    if match:
        return int(match.group(2))
    else:
        return None

def process_csv_dir(directory):
    gpu_data_instances = {}
    cpu_data_instances = {}
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            if 'gpu' in filename:
                gpu_data = pd.read_csv(os.path.join(directory, filename))
                gpu_data = clean_pandas_df(gpu_data)
                gpu_data_instances[instance_number(filename)] = gpu_data
            elif 'cpu' in filename:
                cpu_data = pd.read_csv(os.path.join(directory, filename))
                cpu_data = clean_pandas_df(cpu_data)
                cpu_data_instances[instance_number(filename)] = cpu_data
    return cpu_data_instances, gpu_data_instances

def plot_instance_cpu_gpu(cpu_data, gpu_data, concurrency=1, 
                          variable='Inferences/Second',
                          ylabel='Throughput (infer/sec)',
                          save_name='instances_vs_throughput_compare_con3.pdf'):
    
    plt.figure(figsize=(6, 6))
    
    in_cpu = sorted(cpu_data.keys())
    in_gpu = sorted(gpu_data.keys())
    
    vals_cpu = []
    vals_gpu = []
    for i in in_cpu:
        val = cpu_data[i][cpu_data[i]['Concurrency'] == concurrency][variable].values
        vals_cpu.append(val[0])
    for i in in_gpu:
        val = gpu_data[i][gpu_data[i]['Concurrency'] == concurrency][variable].values
        vals_gpu.append(val[0])
        
    plt.plot(in_cpu, vals_cpu, label='CPU', marker='o')
    plt.plot(in_gpu, vals_gpu, label='GPU', marker='s')
    plt.xlabel('Number of Instances', loc='right')
    plt.ylabel(f'{ylabel} for {concurrency} concurrent requests', loc='top')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f'{args.output_directory}/{save_name}', bbox_inches='tight')

    
def plot_var_vs_instance(data_dict, 
                         variable='Inferences/Second', 
                         ylabel='GPU Throughput (infer/sec)',
                         save_name='instances_vs_throughput_gpu.pdf'):
    
    instances = sorted(data_dict.keys())
    concurrencies = data_dict[1]['Concurrency'].values
    
    plt.figure(figsize=(5, 5))
    for con in concurrencies:
        vals = []
        for i in instances:
            val = data_dict[i][data_dict[i]['Concurrency'] == con][variable].values
            vals.append(val[0])
        plt.plot(instances, vals, 'o-', label=f'{con} concurrent requests')
    plt.xlabel('Number of Instances', loc='right')
    plt.ylabel(ylabel, loc='top')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f'{args.output_directory}/{save_name}', bbox_inches='tight')

def main():
    
    cpu_data_instances, gpu_data_instances = process_csv_dir(args.input_directory)
    
    plot_var_vs_instance(gpu_data_instances)
    plot_var_vs_instance(cpu_data_instances, 
                         ylabel='CPU Throughput (infer/sec)',
                         save_name='instances_vs_throughput_cpu.pdf')
    plot_var_vs_instance(gpu_data_instances, 
                         variable='Avg latency', 
                         ylabel='GPU Latency (us)',
                         save_name='instances_vs_latency_gpu.pdf')
    plot_var_vs_instance(cpu_data_instances,
                         variable='Avg latency',
                         ylabel='CPU Latency (us)',
                         save_name='instances_vs_latency_cpu.pdf')
    plot_var_vs_instance(gpu_data_instances,
                         variable='total_gpu_usage',
                         ylabel='GPU Utilization (%)',
                         save_name='instances_vs_gpu_utilization.pdf')
    
    for i in range(1, 6):
        plot_instance_cpu_gpu(cpu_data_instances, gpu_data_instances, concurrency=i,
                            save_name=f'instances_vs_throughput_compare_con{i}.pdf')
        plot_instance_cpu_gpu(cpu_data_instances, gpu_data_instances, 
                            concurrency=4,
                            variable='Avg latency',
                            ylabel='Average Latency (us)',
                            save_name=f'instances_vs_latency_compare_con{i}.pdf')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-directory", 
                        default='data/instances',
                        type=str, help="Input directory path")
    parser.add_argument("-o", "--output-directory", 
                        default='plots',
                        type=str, help="Output directory path")
    args = parser.parse_args()
    
    main()