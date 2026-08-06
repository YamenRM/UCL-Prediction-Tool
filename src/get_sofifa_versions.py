import pandas as pd
import soccerdata as sd

SEASONS = ['2122', '2223', '2324', '2425', '2526', '2627']

def get_relevant_version_ids(seasons: list[str]) -> list[int]:
    sofifa = sd.SoFIFA(leagues=['ENG-Premier League', 'ESP-La Liga', 'FRA-Ligue 1', 'GER-Bundesliga', 'ITA-Serie A'])
    versions = sofifa.read_versions().reset_index()
    versions['update'] = pd.to_datetime(versions['update'], errors='coerce')

    start_year = 2000 + int(seasons[0][:2])
    range_start = pd.Timestamp(year=start_year, month=6, day=1)
    range_end = pd.Timestamp.today() + pd.Timedelta(days=30)

    mask = versions['update'].between(range_start, range_end)
    relevant = versions[mask].sort_values('update')

    print(f"Found {len(relevant)} relevant FIFA versions out of {len(versions)} total:")
    print(relevant[['version_id', 'fifa_edition', 'update']].to_string(index=False))

    return relevant['version_id'].tolist()


if __name__ == "__main__":
    version_ids = get_relevant_version_ids(SEASONS)
    print(f"\nversion_ids = {version_ids}")