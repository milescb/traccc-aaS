import pandas as pd
import json
import argparse

def main():
    # Read the CSV file
    csv_file_path = args.input
    df = pd.read_csv(csv_file_path)
    
    # make two dfs for geometry id and rest of the features
    geo_id = df['geometry_id'].tolist()
    cells = df.drop('geometry_id', axis=1).values.flatten().tolist()

    # Define JSON structure
    json_data = {
        "data": [
            {
                "GEOMETRY_ID": {
                    "content": geo_id,
                    "shape": [len(df)]
                },
                "FEATURES": {
                    "content": cells,
                    "shape": [len(df), len(df.columns)-1]
                }
            }
        ]
    }

    # Write the JSON data to a file
    json_file_path = args.output
    with open(json_file_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", 
                        default='/global/cfs/projectdirs/m3443/data/traccc-aaS/data/tml_pixels/event000000000-cells.csv',
                        type=str, help="Input CSV file path")
    parser.add_argument("-o", "--output", 
                        default='/global/cfs/projectdirs/m3443/data/traccc-aaS/test_data/test_perf_data.json', 
                        type=str, help="Output JSON file path")
    args = parser.parse_args()

    main()