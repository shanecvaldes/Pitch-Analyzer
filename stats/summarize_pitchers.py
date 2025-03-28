# summarize the pitches from each pitcher and their heatmaps for each pitcher
import os
import pandas as pd
import numpy as np
import csv

def summarize_pitchers_rates():
    # !description, !bb_type, events, type
    columns_stuff = ['game_year', 'pitch_type', 'events', 'type']
    os.makedirs('./pitchers_rates_summarized/', exist_ok=True)
    os.makedirs('./pitches_rates_summarized/', exist_ok=True)
    resulting_cols = ['strike%', 'ball%', 'contact%', 'out%']

    for pitcher in os.listdir(f'./pitchers'):
        print(pitcher)
        df = pd.read_csv(f'./pitchers/{pitcher}', low_memory=False)
        df = df[columns_stuff]
        
        df = df[~df['pitch_type'].isin(['IN', 'PO'])]

        # Count total pitches per (year, pitch type)
        total_pitches = df.groupby(['game_year', 'pitch_type']).size()

        # Count occurrences of specific events
        strike_counts = df[df['type'] == 'S'].groupby(['game_year', 'pitch_type']).size()
        ball_counts = df[df['type'] == 'B'].groupby(['game_year', 'pitch_type']).size()
        contact_counts = df[df['events'].notna()].groupby(['game_year', 'pitch_type']).size()
        out_counts = df[df['events'] == 'field_out'].groupby(['game_year', 'pitch_type']).size()

        # print(out_counts)

        '''summary = pd.DataFrame({
            'pitch_ct': total_pitches,
            'strike%': (strike_counts / total_pitches * 100).fillna(0),
            'ball%': (ball_counts / total_pitches * 100).fillna(0),
            'contact%': (contact_counts / total_pitches * 100).fillna(0),
            'out%': (out_counts / contact_counts * 100)
        })'''

        summary = pd.DataFrame({
            'pitch_ct': total_pitches,
            'strike%': (strike_counts / total_pitches * 100),
            'ball%': (ball_counts / total_pitches * 100),
            'contact%': (contact_counts / total_pitches * 100),
            'out%': (out_counts / contact_counts * 100)
        })

        summary = summary.fillna(0).reset_index()

        # summary = summary.dropna().reset_index()
        # print(len(summary)) 

        summary.to_csv(f'./pitchers_rates_summarized/{pitcher}', index=False)

        # go through the data and insert the mean stats into a csv file grouped by pitch type
        for _, data in summary.iterrows():
            cols = data.keys().tolist()
            data = data.tolist()

            # replace pitch type with pitcher
            pitch_type = data[1]
            data[1] = pitcher.removesuffix('.csv')
            cols[1] = 'pitcher'
            file_name = f'./pitches_rates_summarized/{pitch_type}.csv'

            # if the file does not exist, create it with the new cols
            if not os.path.exists(file_name):
                f = open(file_name, 'w', newline='')
                writer = csv.writer(f)
                writer.writerow(cols)
            with open(file_name, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data)



def summarize_pitchers():
    # spin_axis may/can be used possibly
    columns_stuff = ['game_year', 'pitch_type', 'release_speed', 'pfx_x', 'pfx_z', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'release_spin_rate', 'effective_speed']
    other = ['IN', 'PO']
    os.makedirs('./pitchers_stuff_summarized/', exist_ok=True)
    os.makedirs('./pitches_stuff_summarized/', exist_ok=True)

    # go through each pitcher and take the mean of all the selected statistics, insert into their own csv file
    for pitcher in os.listdir(f'./pitchers'):
        df = pd.read_csv(f'./pitchers/{pitcher}', low_memory=False)

        df = df[columns_stuff]
        '''for column in df.columns:
            if column not in columns_stuff:
                df = df.drop(column, axis=1)'''
        df = df.dropna()

        df = df[~df['pitch_type'].isin(['IN', 'PO'])]

        df = df.groupby(['game_year', 'pitch_type']).mean().reset_index()
        df.to_csv(f'./pitchers_stuff_summarized/{pitcher.removesuffix('.csv')}.csv')

        # go through the data and insert the mean stats into a csv file grouped by pitch type
        for _, data in df.iterrows():
            cols = data.keys().tolist()
            data = data.tolist()

            # replace pitch type with pitcher
            pitch_type = data[1]
            data[1] = pitcher.removesuffix('.csv')
            cols[1] = 'pitcher'
            file_name = f'./pitches_stuff_summarized/{pitch_type}.csv'

            # if the file does not exist, create it with the new cols
            if not os.path.exists(file_name):
                f = open(file_name, 'w', newline='')
                writer = csv.writer(f)
                writer.writerow(cols)
            with open(file_name, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data)


def wipe_folders():
    pass

def main():
    # wipe_folders()
    # summarize_pitchers()
    summarize_pitchers_rates()
    

if __name__ == '__main__':
    main()