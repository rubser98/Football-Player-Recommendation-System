from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import ElementClickInterceptedException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException
import time
from utils import writeJson
import os

def refuseCookies(driver):
    try: 
        cookies = driver.find_element(By.ID, "onetrust-reject-all-handler")
        cookies.click()       
    except:
        pass

def showMore(driver):
    show_more_matches = True
    refuseCookies(driver)
    while show_more_matches:
        try:
            button = driver.find_elements(By.CLASS_NAME, "event__more" )
            if len(button) == 1:
                button = button[0]
            else:
                print(len(button))
                button = None
            if button == None:
                show_more_matches = False
            else:
                #print(button.get_attribute('href'), button.text)
                actions = ActionChains(driver)
                actions.move_to_element(button).perform()
                time.sleep(1)
                #driver.execute_script("arguments[0].scrollIntoView(true);", button)
                actions.click(button).perform()
        
        except StaleElementReferenceException:
            show_more_matches = False

        except NoSuchElementException:
            show_more_matches = False

def getComments(url_commento: str):
    op = webdriver.ChromeOptions()
    op.add_argument('headless')
    driver_match = webdriver.Chrome()
    driver_match.get(url_commento)
    refuseCookies(driver_match)
    teams = [x.text for x in driver_match.find_elements(By.CLASS_NAME, 'participant__participantName')]
    tags = driver_match.find_elements(By.TAG_NAME, 'a')
    day = [x.text for x in tags if 'GIORNATA' in x.text]
    if(len(day) > 0):
        day = day[0].split(' ')[-1]
    else:
        day = 'xx'
        
    if len(teams) == 4:
        home, away = teams[0], teams[2]
    else:
        home, away = teams[0], teams[1]

    home = home.replace('/','_')
    away = away.replace('/','_')

    try:
        #comments_button = [x for x in driver_match.find_elements(By.CLASS_NAME, '_tab_i2rza_4') if x.text == 'COMMENTO'][0]
        comments_button = [x for x in driver_match.find_elements(By.CLASS_NAME, '_tab_myv7u_4') if x.text == 'COMMENTO'][0]
        actions = ActionChains(driver_match)
        actions.move_to_element(comments_button).perform()
        time.sleep(0.05)
                #driver.execute_script("arguments[0].scrollIntoView(true);", button)
        actions.click(comments_button).perform()
        #comments_button.click()
        time.sleep(1)
        comments = [x.text for x in driver_match.find_elements(By.CLASS_NAME, '_commentary_1u4cv_4')]
        driver_match.quit()
        return day, home, away, comments
    
    except Exception as e:
        print(f'{home}-{away}, {e}')
        driver_match.quit()
        return None
    
def mainExtractionComments(url: str, out_path: str):
    driver = webdriver.Chrome()
    driver.get(url)
    showMore(driver)
    matches = driver.find_elements(By.CLASS_NAME, 'eventRowLink')
    matches_urls = [x.get_property('href') for x in matches]
    driver.quit()
    for match in matches_urls:
        comments = getComments(match)
        if comments != None:
            day, home, away, comment_def = comments
            file_out = os.path.join(out_path, f'{day}_{home}_{away}.json')
            writeJson(comment_def, file_out)

def isGoal(commento):
    text = ''
    try:
        badge_info = commento.find_element(By.CLASS_NAME, '_incidentBadge_1u823_4').find_element(By.CLASS_NAME, '_icon_18bay_4')
        risultato = badge_info.text
        print(risultato)
        badge_icon = badge_info.get_attribute('data-testid')
        if badge_icon == 'wcl-icon-incidents-goal-soccer':
            text += f'GOAL! Risultato sul {risultato}. '
        return text
    except:
        return text
    

if __name__ == '__main__':
    seasons = ['2023-2024', '2022-2023', '2021-2022', '2020-2021', '2019-2020']
    leagues = {'italia': ['Serie A'],'inghilterra': ['Premier League'], 'germania': ['Bundesliga'], 'francia': ['Ligue 1'], 'spagna': ['LaLiga'], 'europa':['Champions League', 'Europa League'],'olanda': ['Eredivisie']}

    dataset_path = "Dataset//Direttait//"
    for s in seasons:
        for nat, league in leagues.items():
            for l in league:
                l_url = l.replace(' ', '-').lower()
                url = f'https://www.diretta.it/calcio/{nat}/{l_url}-{s}/risultati/'
                print(f'start {l} {s}')
                league_path = os.path.join(dataset_path, l)
                if not os.path.exists(league_path):
                    os.makedirs(league_path)
                season_league_path = os.path.join(league_path, s)
                if not os.path.exists(season_league_path):
                    os.makedirs(season_league_path)
                mainExtractionComments(url, season_league_path)
                print(f'end {l} {s}')