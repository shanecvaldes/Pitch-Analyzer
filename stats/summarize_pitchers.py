# summarize the pitches from each pitcher and their heatmaps for each pitcher
import os
import pandas as pd
import numpy as np
import csv
def summarize_pitchers():
    # spin_axis may/can be used possibly
    columns_stuff = ['game_year', 'pitch_type', 'release_speed', 'pfx_x', 'pfx_z', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'release_spin_rate', 'effective_speed']
    other = ['IN', 'PO']
    os.makedirs('/pitchers_stuff_summarized/', exist_ok=True)
    os.makedirs('/pitches_stuff_summarized/', exist_ok=True)

    # go through each pitcher and take the mean of all the selected statistics, insert into their own csv file
    for pitcher in os.listdir(f'./pitchers'):
        df = pd.read_csv(f'./pitchers/{pitcher}', low_memory=False)
        for column in df.columns:
            if column not in columns_stuff:
                df = df.drop(column, axis=1)
        df = df.dropna()
        df = df[df.pitch_type != 'IN']
        df = df[df.pitch_type != 'PO']
        df = df.groupby(['game_year', 'pitch_type']).mean().reset_index()
        df.to_csv(f'./pitchers_stuff_summarized/{pitcher.removesuffix('.csv')}_summarized.csv')

        # go through the data and insert the mean stats into a csv file grouped by pitch type
        for _, data in df.iterrows():
            cols = data.keys().tolist()
            data = data.tolist()

            # replace pitch type with pitcher
            pitch_type = data[1]
            data[1] = pitcher.removesuffix('.csv')
            cols[1] = 'pitcher'
            file_name = f'./pitches_stuff_summarized/stuff_summarized_{pitch_type}.csv'

            # if the file does not exist, create it with the new cols
            if not os.path.exists(file_name):
                f = open(file_name, 'w', newline='')
                writer = csv.writer(f)
                writer.writerow(cols)
            with open(file_name, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data)


def main():
    summarize_pitchers()

if __name__ == '__main__':
    main()