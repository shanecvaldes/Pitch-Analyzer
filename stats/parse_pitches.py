# summarize the pitches from each pitcher and their heatmaps for each pitcher
import os
import pandas as pd
import numpy as np
import csv

# put all the pitches into their own files by pitch type
def parse_pitches():
    os.makedirs('/pitches/', exist_ok=True)
    for pitcher in os.listdir(f'./pitchers'):
        print(pitcher)
        df = pd.read_csv(f'./pitchers/{pitcher}', low_memory=False)
        df = df[df['pitch_type'].notna()]

        df = df.groupby(['pitch_type'])
        
        # Go through the data grouped by the pitch type and insert into a csv file
        # create the csv file if it doesn't already exist
        for pitch, data in df:
            cols = data.keys().tolist()
            file_name = f'./pitches/all_{pitch[0]}.csv'

            if not os.path.exists(file_name):
                f = open(file_name, 'w', newline='')
                writer = csv.writer(f)
                writer.writerow(cols)
            with open(file_name, 'a', newline='') as f:
                writer = csv.writer(f)
                for row in data.values:
                    writer.writerow(row)

def main():
    parse_pitches()

if __name__ == '__main__':
    main()