import pandas as pd
import numpy as np
import utils
import os
from pathlib import Path
import argparse

def getPlayersMatch(df_events):
  players = {}
  for id, player in zip(df_events.playerId, df_events.playerName):
    if not np.isnan(id):
      players[id] = player

  return players

def translateRelatedEvent(satisfiedEvents):
  file = 'Dataset/WhoScored/event_metadata.json'
  event_metadata = utils.readJson(file)
  event_metadata = {value : key for key,value in event_metadata.items()}
  return [event_metadata[x] for x in satisfiedEvents]

def misureDistance(x1, y1, x2, y2):
  return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def traduciEvento(events_dict,event, lang='en'):
  #events_dict = readJson( f'{GDRIVE_THESIS_DIR}/action_translations.json')
  events = events_dict[lang]['events']
  related_events = events_dict[lang]['relatedEvents']
  current_event = event['displayName']
  translated_event = events[current_event]
  #current_related_events = translateRelatedEvent(satisfiedEvents)
  #translated_related_events = [related_events[x] for x in current_related_events if related_events[x] != '']
  #str_related_events = f" ({', '.join(translated_related_events).strip(', ')})" if len(translated_related_events) > 0 else ""
  return translated_event

def traduciEventiSupplementari(events_dict, satisfiedEvents, qualifiers, lang='en'):
  qualifiers_dict = events_dict[lang]['qualifiers']
  related_events = events_dict[lang]['relatedEvents']
  current_related_events = translateRelatedEvent(satisfiedEvents)
  qualifierNames = [x['type']['displayName'] for x in qualifiers]
  translated_qualifiers = [qualifiers_dict[x] for x in qualifierNames if qualifiers_dict[x] != '']
  translated_related_events = [related_events[x] for x in current_related_events if related_events[x] != ''] + translated_qualifiers
  str_related_events = f" ({', '.join(translated_related_events).strip(', ')})" if len(translated_related_events) > 0 else ""

  return str_related_events


def componiFrase(row, events_dict, lang='en'):
    #esito = 'con successo' if row['outcomeType']['value'] == 1 else 'fallendo'
    start_position = detectFieldBin(row['x'], row['y'])
    prep_from = 'da' if lang == 'it' else 'from'
    prep_to = 'a' if lang == 'it' else 'to'
    if start_position != '':

      start_position = f"{prep_from} {start_position}"
    else:
      start_position = ''

    if np.isnan(row['endX']) or np.isnan(row['endY']):
      end_position = ''
    else:
      end_position = f"{prep_to} {detectFieldBin(row['endX'], row['endY'])}"

    frase = f"{traduciEvento(events_dict, row['type'], lang)} {start_position} {end_position} {traduciEventiSupplementari(events_dict, row['satisfiedEventsTypes'], row['qualifiers'], lang)}"
    return frase.strip()

def detectFieldBin(x , y, n_binx: int = 5, n_biny: int = 5, lang: str = 'en'):
    if x == 0 and y == 0:
      return ''
    if lang == 'it':
      x_text = {1: 'difesa', 2: 'trequarti difensiva', 3: 'centrocampo', 4: 'trequarti offensiva', 5: 'attacco'}
      y_text = {1: 'fascia destra', 2: 'centro destra', 3: 'centrale', 4: 'centro sinistra', 5: 'fascia sinistra'}
    else:
      x_text = {1: 'defensive', 2: 'defensive third', 3: 'midfield', 4: 'attacking third', 5: 'attack'}
      y_text = {1: 'right flank', 2: 'right center', 3: 'central', 4: 'left center', 5: 'left flank'}

    field_length_x = 100
    field_length_y = 100
    bin_x_width, bin_y_width = np.ceil(field_length_x / n_binx), np.ceil(field_length_y / n_biny)
    bin_x = int((x - 1) / bin_x_width) + 1
    bin_y = int((y - 1) / bin_y_width) + 1
    return f"{x_text[bin_x]}, {y_text[bin_y]}"

def componiTextPlayerMatch(player_match: dict) -> str:
    text = ''
    i = 0
    for id, action in player_match['text'].items():
        if i > 0:
            if int(id) == i+1:
                text = text +', ' + action
            else:
                text = text +'. ' + action
        else:
            text = action
        i = int(id)

    return text

def mainCreazioneDocumenti(src_dir, lang):

    events_dict = utils.readJson( f'action_translations.json')
    src_dir_path = Path(src_dir)
    local_src_dir = src_dir.split('/')[-1]
    src_dir_league = [d.name for d in src_dir_path.iterdir() if d.is_dir() and '-' in d.name]
    for league_dir in src_dir_league:
        league_dir_abs = os.path.join(src_dir, league_dir)
        tgt_league_dir_abs = league_dir_abs.replace(local_src_dir, f'Events2Text_{lang}')
        os.makedirs(tgt_league_dir_abs, exist_ok=True)
        league_dir_path = Path(league_dir_abs)
        src_dir_league_season = [d.name for d in league_dir_path.iterdir() if d.is_dir() and '-' in d.name]
        for season in src_dir_league_season:
            src_dir_full = os.path.join(league_dir_abs, season)
            tgt_dir_full = os.path.join(tgt_league_dir_abs,season)
            os.makedirs(tgt_dir_full, exist_ok=True)
            input_files = os.listdir(src_dir_full)
            for match in input_files:
                match_events = []
                file = os.path.join(src_dir_full, match)
                df_events = pd.read_json(file)

                for idx, row in df_events.iterrows():
                    match_events.append({'playerId': row['playerId'], 'playerName': row['playerName'],'teamId': row['teamId'], 'team': row['teamName'], 'text': componiFrase(row, events_dict, lang)})
                    
                utils.writeJson(match_events, os.path.join(tgt_dir_full, match))


def getPlayersMatch(df_events):
    teams = {}
    try:
        for id, name in zip(df_events.teamId, df_events.teamName):
            if not pd.isna(name):
                teams[id] = name
    except:
        for id, name in zip(df_events.teamId, df_events.team):
            if not pd.isna(name):
                teams[id] = name

    teams_list = {x: { 'name': teams[x], 'players': {}} for x in teams.keys()}

    for id, row in df_events.iterrows():
        teamId = row['teamId']
        id = row['playerId']
        if not np.isnan(id):
            teams_list[teamId]['players'][id] = row['playerName']

    return teams_list

def aggiungiPlayersFromMatch(teams_list_all, teams_list_match):
    for team in teams_list_match.keys():
        if team not in teams_list_all:
            teams_list_all[team] = teams_list_match[team]
        else:
            teams_list_all[team]['players'] = teams_list_all[team]['players'] | teams_list_match[team]['players'] 

#utilizzata per creare file con squadre-giocatori per stagioni utilizzata successivamente per lista giocatori con modelli
def getPlayersTeamListPerSeason(src_dir, tgt_dir):
    src_dir_path = Path(src_dir)
    src_dir_league = [d.name for d in src_dir_path.iterdir() if d.is_dir() and '-' in d.name]
    for league_dir in src_dir_league:
        league_dir_abs = os.path.join(src_dir, league_dir)
        league_dir_path = Path(league_dir_abs)
        src_dir_league_season = [d.name for d in league_dir_path.iterdir() if d.is_dir() and '-' in d.name]
        for season in src_dir_league_season:
            src_dir_full = os.path.join(league_dir_abs, season)
            input_files = os.listdir(src_dir_full)
            teams_list = {}
            for match in input_files:
                file = os.path.join(src_dir_full, match)
                df_events = pd.read_json(file)
                match_players = getPlayersMatch(df_events)
                aggiungiPlayersFromMatch(teams_list, match_players)
            
            utils.writeJson(teams_list, os.path.join(tgt_dir, f'{league_dir}_{season}_teams.json'))
            
def creaDatasetFinale(src_dir, lang):
    entire_dataset = []
    for x in os.listdir(src_dir):
        league_dir = f'{src_dir}/{x}' 
        if os.path.isdir(league_dir) and '-' in x:
            for season in os.listdir(league_dir):
                league_season_dir = f'{league_dir}/{season}' 
                print(league_season_dir)
                for match_json in os.listdir(league_season_dir):
                    match = match_json[:-5]
                    df_match = pd.read_json(f'{league_season_dir}/{match_json}')
                    players_team = getPlayersMatch(df_match)
                    for team in players_team.keys():
                        players = players_team[team]['players']
                        for p in players.keys():
                            df_match_player = df_match[df_match.playerId == p]
                            text = componiTextPlayerMatch(df_match_player)
                            teamName = players_team[team]['name']
                            playerName = players[p]
                            record = dict(season=season, playerId=p, playerName=playerName, teamId=team, teamName=teamName, text=text, match=match)
                            entire_dataset.append(record)

    print(f'Numero di record caricati: {len(entire_dataset)}')
    utils.writeJson(entire_dataset, f'{src_dir}/player2vec_dataset_{lang}.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate player and team descriptions")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the directory containing the dataset file.")
    parser.add_argument("--lang", type=str, required=True, choices=["it", "en"], help="Language option. Choose between 'it' (Italian) or 'en' (English).")
    args = parser.parse_args()

    mainCreazioneDocumenti(args.dataset_dir, args.lang)
    txt_dir = f'Events2Text_{args.lang}'
    creaDatasetFinale(txt_dir, args.lang)
