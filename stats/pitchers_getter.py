import asyncio
from bs4 import BeautifulSoup
import aiohttp
import aiofiles
import os
import pandas as pd
import csv
async def download_all(session, player_ids):
    main_url = 'https://baseballsavant.mlb.com/statcast_search/csv?hfPT=&hfAB=&hfGT=R%7C&hfPR=&hfZ=&hfStadium=&hfBBL=&hfNewZones=&hfPull=&hfC=&hfSea=2024%7C2023%7C2022%7C2021%7C2020%7C2019%7C2018%7C2017%7C2016%7C2015%7C2014%7C2013%7C2012%7C2011%7C2010%7C2009%7C2008%7C&hfSit=&player_type=pitcher&hfOuts=&hfOpponent=&pitcher_throws=&batter_stands=&hfSA=&game_date_gt=&game_date_lt=&hfMo=&hfTeam=&home_road=&hfRO=&position=&hfInfield=&hfOutfield=&hfInn=&hfBBT=&hfFlag=&metric_1=&group_by=name&min_pitches=0&min_results=0&min_pas=0&sort_col=pitches&player_event_sort=api_p_release_speed&sort_order=desc&type=details&player_id={id}&minors=false'
    # Pat the unicorn causes trouble
    for id, name in player_ids:
        # overwrite the Pat Venditte, unicorn emoji is unreadable in some cases
        if id == '519381':
            name = 'Venditte, Pat RLHP'
        async with session.get(main_url.format(id=id)) as response:
            async with aiofiles.open(f'pitchers/{name}.csv', mode='wb') as f:
                await f.write(await response.read())
                try:
                    df = pd.read_csv(f'./pitchers/{name}.csv', low_memory=False)
                except:
                    # try again
                    os.remove(f'./pitchers/{name}')
                # sometimes the response is an html file
        # await asyncio.sleep(1)

async def get_players(session):
    url = 'https://baseballsavant.mlb.com/statcast_search?hfPT=&hfAB=&hfGT=R%7C&hfPR=&hfZ=&hfStadium=&hfBBL=&hfNewZones=&hfPull=&hfC=&hfSea=2024%7C2023%7C2022%7C2021%7C2020%7C2019%7C2018%7C2017%7C2016%7C2015%7C2014%7C2013%7C2012%7C2011%7C2010%7C2009%7C2008%7C&hfSit=&player_type=pitcher&hfOuts=&hfOpponent=&pitcher_throws=&batter_stands=&hfSA=&game_date_gt=&game_date_lt=&hfMo=&hfTeam=&home_road=&hfRO=&position=&hfInfield=&hfOutfield=&hfInn=&hfBBT=&hfFlag=&metric_1=&group_by=name&min_pitches=0&min_results=0&min_pas=0&sort_col=pitches&player_event_sort=api_p_release_speed&sort_order=desc#results'
    async with session.get(url) as response:
        markdown = BeautifulSoup(await response.read(), features='html.parser')
        markdown.prettify()
        rows = markdown.find_all('tr')
        result = []
        # go through each table row of players, add to the download queue only if the pitcher data is not already in the folder
        for row in rows:
            if row.get('id') is not None and len(row.find_all('td', {'class':'player_name'})) > 0:
                player_name = row.find_all('td', {'class':'player_name'})[0].text.strip()
                player_id = row.get('id').removeprefix('player_name_')[:-1]

                # make sure that there are no duplicates
                if player_id == '519381':
                    player_name = 'Venditte, Pat RLHP'
                if player_name not in [i.removesuffix('.csv') for i in os.listdir('./pitchers')]:
                    print(player_name, player_id)
                    result.append((player_id, player_name))
        return result

        # return [(,  for i in rows[::2] if i.get('id') is not None]




async def download():

    async with aiohttp.ClientSession() as session:
        player_ids = await get_players(session)
        # print(player_ids, len(player_ids))
        while len(player_ids) > 0:
            await download_all(session, player_ids)
            player_ids = await get_players(session)
    print('Download completed')

    pass

# warning: the resulting csv file is too large to really manage by itself, keep for reference
async def merge_all():

    # combined_df = None
    failures = []
    flag = True
    with open('combined_pitches.csv', 'w') as f:
        writer = csv.writer(f)
        # df = pd.read_csv(f'./pitches/{file}', low_memory=False).keys()

        for file in os.listdir('./pitchers'):
            print(file)
            try:
                df = pd.read_csv(f'./pitchers/{file}', low_memory=False)
            except:
                # try again
                os.remove(f'./pitchers/{file}')
            if flag:
                cols = df.keys().tolist()
                writer.writerow(cols)
                flag = False

            for row in df.values.tolist():
                writer.writerow(row)
        await download()
        for failure in failures:
            df = pd.read_csv(failure, low_memory=False)
            for row in df.values.tolist():
                writer.writerow(row)
    

async def main():
    data = await download()
    # await merge_all()
    

if __name__ == '__main__':
    asyncio.run(main())