from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import ElementClickInterceptedException
from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException
import time
from utils import writeJson, readJson
import os
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import json
import re
from bs4 import BeautifulSoup as soup
import datetime
from datetime import datetime as dt
from tqdm import tqdm

def getPlayerStats(driver):
    data = {}
    tables = driver.find_elements(By.CLASS_NAME, 'stats_table')
    table = None
    for t in tables:
        if "scout_full" in t.get_attribute("id"):
            table = t
            break
    tags = ['<br>', '<strong>', '</strong>']
    type_stat = 'Standard Stats'
    data[type_stat] = {}
    for row in table.find_elements(By.TAG_NAME, 'tr'):
        th = row.find_element(By.TAG_NAME, 'th')
        #print(f'--{row.get_property("className")} -- {row.text}')
        if row.get_property("className") == "thead over_header thead":
            type_stat = th.text
            data[type_stat] = {}
            #print(type_stat)
        
        data_desc = th.get_attribute('data-tip')
        

        tds = row.find_elements(By.TAG_NAME, 'td')
        if len(tds) > 0 and th.text != '':
            value, perc = tds[0].text , tds[1].text
            #f'{th.text} ({data_desc})'
            for t in tags:
                if data_desc != None:
                    data_desc = data_desc.replace(t, ' ')
                
            data[type_stat][th.text] = {'description': data_desc, 'value': value, 'percentile': perc.strip()}

    return data

def getRoles(role):
    rr=[]
    if '(' in role:
        roles_split = role.split('(')
    else:
        roles_split = [role]

    for r in roles_split:
        if '-' in r:
            r_split = r.split('-')
            rr.append(r_split[0])
            rl = r_split[1]
            if rl[-1] == ')':
                rl = rl[:-1]
            rr.append(rl)

        elif ',' in r:
            rr.append(r.split(',')[0])
        elif ')' in r:
            rr.append(r[:-1])
        else:
            rr.append(r.strip())


    return rr

def initializeDriver():
    driver = webdriver.Chrome()
    url = 'https://fbref.com/en/'
    driver.get(url)
    cookie_button = driver.find_elements(By.TAG_NAME, 'button')
    for b in cookie_button:
        if b.text == 'Accetta tutto':
            b.click()
    return driver

def getPlayerAnag(driver, player_dict):
    try:
        more_button = driver.find_element(By.XPATH,'//*[@id="meta_more_button"]')
        more_button.click()
    except:
        pass

    #anag_div = driver.find_element(By.XPATH, '/html/body/div[4]/div[3]/div[1]/div[2]')
    #player = anag_div.find_element(By.TAG_NAME, 'h1').text
    i=1
    anag_elem1= ''

    try:
        driver.find_element(By.XPATH,f'//*[@id="meta"]/div[2]')
        tab_prefix = '//*[@id="meta"]/div[2]'
    except:
        tab_prefix = '//*[@id="meta"]/div'
    
    while not anag_elem1.startswith("Position"):
        anag_elem1 = driver.find_element(By.XPATH,f'{tab_prefix}/p[{i}]').text
        i+=1
        if i==10:
            raise KeyError
    #anag_elem1 = driver.find_element(By.XPATH,'//*[@id="meta"]/div[2]/p[1]').text
    anag_elem2 = driver.find_element(By.XPATH,f'{tab_prefix}/p[{i}]').text
    anag_elem3 = driver.find_element(By.XPATH,f'{tab_prefix}/p[{i+1}]').text
    #anag_elem4 = driver.find_element(By.XPATH,f'{tab_prefix}/p[{i+2}]').text
    '''
    try:
        anag_elem5 = driver.find_element(By.XPATH,f'//*[@id="meta"]/div[2]/p[{i+3}]').text
    except:
        anag_elem5 = ''
    '''
    anag_elem1_split = anag_elem1.split('▪')
    #anag_elem2_split = anag_elem2.split(' ')
    position = anag_elem1_split[0].split(':')[1].strip()
    #footed = anag_elem1_split[1].split(':')[1].strip()
    if anag_elem2.startswith('Born'):
        year_birth = anag_elem2.split(' ')[3]
        #height = ''
        #weight = ''
        nat = anag_elem3.split(' ')[2] if anag_elem3.startswith('National Team') else anag_elem3.split(' ')[1]
        #team = ' '.join(anag_elem5.split(' ')[1:]) if anag_elem4 != '' else ''

    else:
        #height, weight = anag_elem2_split[0].replace(',',''), anag_elem2_split[1]
        year_birth = anag_elem3.split(' ')[3]
        #nat = anag_elem4.split(' ')[2] if anag_elem3.startswith('National Team') else anag_elem3.split(' ')[1]
        #team = ' '.join(anag_elem5.split(' ')[1:]) if anag_elem5 != '' else ''
        
    addict = dict(#player=player, 
                position=position 
                #foot=footed, 
                #height= height, 
                #weight=weight, 
                ,year_birth=year_birth
                #,nationality=nat
                #,team=team
                )

    return player_dict | addict


def getPlayerRecord(driver, player_dict):
    url = player_dict['link']
    driver.get(url)
    time.sleep(0.5)
    try:
        stats = getPlayerStats(driver)
        player_dict['stats'] = stats
    except:
        pass
    if 'stats' in player_dict.keys():
        player_dict = getPlayerAnag(driver, player_dict)
    else:
        player_dict['stats'] = {}
    
    
    return player_dict


def getTeamPlayers(driver, url : str, id_league: int, team: str) -> list:
    urls=[]
    driver.get(url)
    #id_league = '12229'
    #table = driver.find_element(By.XPATH, '//*[@id="stats_standard_11"]/tbody')
    table = driver.find_element(By.CLASS_NAME, 'stats_table')
    
    rows = table.find_elements(By.TAG_NAME, 'tr')
    for r in rows:
        presenze = '0'
        th = r.find_element(By.TAG_NAME, 'th')
        if th.get_attribute('csk') != None:
            tds = r.find_elements(By.TAG_NAME, 'td')
            for td in tds:
                if td.get_attribute('data-stat') == 'games':
                    presenze = td.text
            if int(presenze) > 5:
                player_name = th.find_element(By.TAG_NAME, 'a').get_attribute('href').split('/')[-1].replace('-',' ')
                #player_name = th.text
                id_player = th.get_attribute('data-append-csv')
                #print(th.text,th.get_attribute('data-append-csv'), presenze)
                url_p=f"https://fbref.com/en/players/{id_player}/scout/{id_league}/{player_name.replace(' ', '-')}-Scouting-Report"
                urls.append(dict(id=id_player, name=player_name, link =url_p, team=team))
    return urls

def getLeagueTeams(driver, url, id):
    driver.get(url)
    team_links=[]
    #table = driver.find_element(By.XPATH, '//*[@id="results2023-202491_overall"]/tbody')
    #table = driver.find_element(By.TAG_NAME, 'tbody')
    table = driver.find_element(By.CLASS_NAME, 'stats_table')
    table = table.find_element(By.TAG_NAME, 'tbody')
    #print(table.text)
    rows = table.find_elements(By.TAG_NAME, 'tr')
    for r in rows:
        l = r.find_element(By.TAG_NAME,'a')
        team = l.text
        link= l.get_property('href')
        team_links.append(dict(id=id, team=team, link=link))
    return team_links


def estrazioneSquadre():
    leagues = readJson('Dataset/Fbref/competitions.json')
    team_leagues = []
    driver=initializeDriver()
    for id, link in leagues.items():
        print(id,link)
        teams = getLeagueTeams(driver, link, id)
        team_leagues= team_leagues + teams

    writeJson(team_leagues, 'Dataset/Fbref/teams.json')


def estrazioneLinkGiocatori():
    team_leagues = readJson('Dataset/Fbref/teams.json')
    player_links= []
    driver=initializeDriver()
    with tqdm(total=len(team_leagues), desc="Extracting players link") as pbar:
        for team in team_leagues:
            players_team = getTeamPlayers(driver, team['link'], team['id'], team['team'])
            player_links= player_links + players_team
            #print(player_links)
            pbar.update(1)

    driver.quit()
    writeJson(player_links, 'Dataset/Fbref/players.json')

def estrazionePlayerStats():
    players = readJson('Dataset/Fbref/players.json')
    driver = initializeDriver()
    with tqdm(total=len(players), desc="Processing players") as pbar:

        for i in range(len(players)):
            if 'stats' not in players[i].keys():
                players[i] = getPlayerRecord(driver, players[i])
            
            if i %10 == 0:
                writeJson(players, 'Dataset/Fbref/players.json')
            
            pbar.update(1)
    writeJson(players, 'Dataset/Fbref/players.json')
    driver.quit()

def get_stats_quality(perc: int) -> str:
    if perc <= 30:
        return 'Weak'
    elif perc <= 49:
        return 'Average'
    elif perc <= 74:
        return 'Good'
    elif perc <= 89:
        return 'Very good'
    else:
        return 'Excellent'
    
def generatePromptPlayer():
    prompts = {}

    records = readJson('Dataset/transfermarkt_fbref_dataset.json')
    records = [x for x in records if x['stats'] != {} and x['position'] != 'GK']
    tm_pos = readJson('Dataset/tm_position_translation.json')
    stats_dict = {'Shooting':['Shooting'], 'Passing': ['Goal and Shot Creation','Passing', 'Pass Types'], 'Possession': ['Possession'], 'Defensive':['Defense']}#, 'Miscellaneous Stats']}
    for p in records:
        player_name = p['id']#p['name']

        #print(player_name, p['name'], p['link'], p['position'])
        prompts[player_name] = {'prompt':{}}
        #roles = getRoles(p['position'])
        #roles_verb = [position_dict[x.strip()] for x in roles]
        #roles_str = ', '.join(roles_verb)
        role = p['tm_role']
        roles_str = tm_pos[role]
        #stats_dict = p['stats']
        prompts[player_name]['position_str'] = roles_str
        prompts[player_name]['name'] = p['name']
        for k, tab  in list(stats_dict.items())[0:]:
            stat = tab
            stat = {}
            for t in tab:
                stat = stat | p['stats'][t]

            prompt = f""""You are a professional soccer scout with expertise in analyzing players' technical and tactical characteristics.
            ## Task:  
            Generate a very concise description (75 tokens) of a player's {k} performance based on the provided per-match statistics. 
            The analysis should:    
                - Highlight the player's **key strengths**.
                - Identify potential **areas for improvement**.
                - Be **role-specific**, considering the player's position.
            
            ## Reasoning Process:
                - Interpret the player's style of play:
                    Identify how the player's strengths shape their contributions.
                    Explain how weaknesses may limit their effectiveness.
                
                -Generate a natural, fluent description:
                    Highlight key strengths that define the player's ability.
                    Mention secondary strengths if relevant.
                    Point areas for improvement, keeping a constructive tone.
                
                - If data concerns about defense and the player plays in defensive roles (e.g. central back, fullback) use more 100 tokens.
                

            ## Input Format:
                - **Position:** Preferred positions 
                - **Statistics:** Skill performance where it is indicated level and description
            
            ## Output Guidelines:
                - **Professional & concise** language suitable for scouting.
                - **Short** The report MUST be very concise, around 50 tokens, if the position is a defensive role (e.g. Defender, central back, fullback) and data are related to defense use 100 tokens
                - **Plain text only** (no bullet points, code, or structured output).
                - **Do not include raw statistics and percentiles information** (focus on interpretation).
                - **No predictions** about future performance.
                - **Only use provided data**, without speculation.
                - **The description should be role-specific**, considering the player's position.
                - The report regards only one player.
                - The report should be similar to the following example output structure.
                - Don't add new input data
                - [END_REPORT] when you end the description
            
            ## Example Output:
                A well-rounded attacking midfielder with exceptional ability in progressing the ball and creating goal-scoring opportunities. 
                He excels in shot-creating actions, with a strong ability to beat defenders through take-ons. 
                His capacity to contribute directly to goals is elite. 

            ### Input Data:
                - **Position**: {roles_str}
                - **Statistics**:    
            """
            for st, v in stat.items():
                level = f"- {st} (Level: {get_stats_quality(int(v['percentile']))}): {v['description']}\n"
                prompt+= level

            prompt+= "###Generated Report:"
            prompts[player_name]['prompt'][k] = prompt

    writeJson(prompts, 'Descriptions/player_descriptions_v2.json')

def cleanDesc(desc):
    c_square = desc.count('[END_REPORT]')
    c_plain = desc.count('END_REPORT')
    c = max(c_square, c_plain)
    sep = '[END_REPORT]' if c_square > 0 else 'END_REPORT'
    if c == 0:
        sep = "### Input Data:"
    splits = desc.split(sep)          
    splits = [s.strip() for s in splits]
    if splits[0] == '':
        return splits[1]
    else:
        return splits[0]


def getTeamPlayerIds(dataset):
    teams = set(x['team'] for x in dataset)
    dataset = [x for x in dataset if 'position' in x.keys() and x['position'] != 'GK']
    team_player_list = {}
    for t in teams:
        players = []
        for p in dataset:
            if p['team'] == t:
                players.append(p['id'])
        
        team_player_list[t] = players
    
    return team_player_list


def isDefender(p):
    if 'Defender' in p['position_str'] or 'back'in p['position_str']:
        return True
    return False

def generatePromptTeam():
    player_desc = readJson("Descriptions/Descriptions/player_descriptions_v2.json")
    #player_desc = player_desc | readJson("Descriptions/Descriptions/player_descriptions_mancanti.json")
    dataset_in = readJson("Dataset/transfermarkt_fbref_dataset.json")
    dataset_manc = readJson('Dataset/transfermarkt_fbref_mancanti.json')
    #merge giocatori mancanti con dataset originale
    for i in range(len(dataset_manc)):
        for j in range(len(dataset_in)):
            if dataset_manc[i]['stats'] != {} and dataset_manc[i]['id'] == dataset_in[j]['id']:
                dataset_in[j] = dataset_manc[i]

    team_player_list = getTeamPlayerIds(dataset_in)
    team_dict = {}
    for team, ps in team_player_list.items():
        
            team_dict[team] = {}
            team_dict[team]['prompt'] = {}

            prompt=f"""You are a professional soccer analyst specializing in squad-building strategies.
                ## Task:
                Analyze the individual descriptions of a team's players and determine key attributes or tactical elements that are missing from the squad, which could impact overall balance and effectiveness.
                ## Reasoning process:
                1. Group players by position (Defenders, Midfielders, Forwarders) based on the provided descriptions.
                2. Identify the prioritized characteristics by analyzing the most frequently mentioned strengths and weaknesses for each role.
                3. Analyze potential gaps in squad composition by detecting missing traits that could improve balance (e.g., lack of creativity in midfield, absence of aerially dominant forwards, limited pace in defense).
                4. Summarize the findings, highlighting both the team's key characteristics and the notable absences that could impact performance.
                
                ## Input format:
                Team Name: [Team Name]
                Player Descriptions: A list of individual player descriptions, each one providing insights into the player's strengths and weaknesses.
                
                ## Output guidelines:
                Provide a concise (200 tokens) summary of the characteristics valued by the team for each role.
                Ensure the descriptions are coherent and reflect a structured playing philosophy.
                Use professional and analytical language.
                Do not include individual player names—focus on general trends.
                Plain text only (no bullet points, code, or structured output).
                Use [END_REPORT] when you end the description
                The report must have the same structure of the example output
                
                ## Example Input:
                Team Name: FC Example
                Player Descriptions:
                - Defender, Central back: A physical and aggressive center-back who dominates in aerial duels and defensive tackles but struggles in ball progression.
                - Fullback: A dynamic fullback with high stamina and strong defensive positioning, yet limited attacking contributions.
                - Midfielder, Central midfielder: A central midfielder with outstanding pressing ability and quick passing, but lacking goal-scoring instinct.
                - Attacking midfielder: An attacking winger with elite dribbling and acceleration, excelling in 1v1 situations but offering little defensive work.
                - Forwarder: A striker with a clinical finishing ability and strong positioning inside the box, but not very involved in buildup play.
                ## Example Output:
                FC Example builds its squad around a physically dominant defensive structure, center-backs with strong aerial ability and defensive aggression, though they contribute less in possession. Fullbacks are selected for their defensive stability rather than attacking impact. In midfield, the club prioritizes high-intensity pressing and quick ball circulation over goal-scoring ability. Their attacking philosophy emphasizes pace and individual skill, with wingers excelling in 1v1 situations and strikers focused on efficient finishing rather than playmaking.
                
                ## Input data:
                Team Name: {team}
                Player descriptions:
                """
            #ps = list(players.keys())

                
            for p in ps:
                
                desc_keys = list(player_desc[p]['description'].keys())[:-1]
                if isDefender(player_desc[p]):
                    desc_keys = desc_keys[::-1]
                    
                desc = ''.join([player_desc[p]['description'][k] for k in desc_keys])
                #prompt+= f"- {p}: {cleanDesc(players[p]['description']['summary'])}\n"
                prompt+= f"- {player_desc[p]['position_str']}: {desc}\n"
                #prompt+= f"- {players[p]['position']}: {desc}\n"

            prompt+="###Generated Report:"
            #missing_prompt+="###Generated Report:"

            team_dict[team]['prompt']['general'] = prompt
            #team_dict[team]['prompt']['missing'] = missing_prompt

    writeJson(team_dict, 'Descriptions/team_descriptions_v2.json')


if __name__ == '__main__':

    estrazioneSquadre()
    estrazioneLinkGiocatori()
    estrazionePlayerStats()
    generatePromptPlayer()
    generatePromptTeam()

