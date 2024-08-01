import pandas as pd
import json
import argparse
import os

def main():
    # Read the CSV file
    csv_dir = args.input
    
    data = []
    index = 0
    for file in os.listdir(csv_dir):
        if file.endswith("cells.csv"):
            csv_file_path = os.path.join(csv_dir, file)
            df = pd.read_csv(csv_file_path)
            
            # make two dfs for geometry id and rest of the features
            geo_id = df['geometry_id'].tolist()
            cells = df.drop('geometry_id', axis=1).values.flatten().tolist()
            
            iInput = {
                "GEOMETRY_ID": {
                    "content": geo_id,
                    "shape": [len(df)]
                },
                "FEATURES": {
                    "content": cells,
                    "shape": [len(df), len(df.columns)-1]
                }
            }
            
            data.append(iInput)
            
            # if index > 0:
            #     break
            # index += 1
            
    print(len(data))

    # Define JSON structure
    json_data = { "data": data }

    # Write the JSON data to a file
    json_file_path = args.output
    with open(json_file_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=2)
        
        #event000000000-cells.csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", 
                        default='/global/cfs/projectdirs/m3443/data/traccc-aaS/data/tml_pixels/',
                        type=str, help="Input CSV file path")
    parser.add_argument("-o", "--output", 
                        default='/global/cfs/projectdirs/m3443/data/traccc-aaS/test_data/test_perf_data.json', 
                        type=str, help="Output JSON file path")
    args = parser.parse_args()

    main()