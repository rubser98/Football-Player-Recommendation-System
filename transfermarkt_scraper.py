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
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import json

def contieneTitolo(box):
    try:
        box.find_element(By.CLASS_NAME, 'content-box-headline')
        return True
    except:
        return False

def acceptCookie(driver):
    try:
        time.sleep(5)
        iframe = driver.find_element(By.CSS_SELECTOR, "#sp_message_iframe_953827")
        driver.switch_to.frame(iframe)
        button = driver.find_element(By.CSS_SELECTOR, "button")
        button.click()
        time.sleep(5)
    except:
        print('No cookies to accept')

def getTeamsURLS(driver, leagueUrl):
    try:
        driver.get(leagueUrl)
        #acceptCookie(driver)
        box_titolati = [x for x in driver.find_elements(By.CLASS_NAME, 'box') if contieneTitolo(x)]
        box_squadre = box_titolati[1]
        odds = box_squadre.find_elements(By.CLASS_NAME, 'odd')
        evens = box_squadre.find_elements(By.CLASS_NAME, 'even')
        team_urls = []
        for team in odds:
            team_urls.append(team.find_element(By.TAG_NAME, 'a').get_property('href'))
        for team in evens:
            team_urls.append(team.find_element(By.TAG_NAME, 'a').get_property('href'))
        return team_urls
    except Exception as e:
        print(f'Squadre non trovate url {leagueUrl}, {e}')
        return []

def NumberPlayers(driver):
    table = driver.find_element(By.CLASS_NAME,'items')
    trs =table.find_element(By.TAG_NAME, 'tbody').find_elements(By.CLASS_NAME, 'odd')
    trs = trs + table.find_element(By.TAG_NAME, 'tbody').find_elements(By.CLASS_NAME, 'even')
    return len(trs)

def getTeamName(driver):
    return driver.find_element(By.XPATH, '//*[@id="tm-main"]/header/div[1]/h1').text

def getPlayersInfo(table, i, team_name):
    numero_maglia = table.find_element(By.XPATH,f'//*[@id="yw1"]/table/tbody/tr[{i}]/td[1]').text
    player, role = table.find_element(By.XPATH,f'//*[@id="yw1"]/table/tbody/tr[{i}]/td[2]').text.split('\n')
    id = table.find_element(By.XPATH,f'//*[@id="yw1"]/table/tbody/tr[{i}]/td[2]/table/tbody/tr[1]/td[2]/a').get_property('href').split('/')[-1]
    data_nasc = table.find_element(By.XPATH,f'//*[@id="yw1"]/table/tbody/tr[{i}]/td[3]').text.split()[0]
    nat  = table.find_element(By.XPATH,f'//*[@id="yw1"]/table/tbody/tr[{i}]/td[4]/img[1]').get_property('title')
    #team = table.find_element(By.XPATH, f'//*[@id="yw1"]/table/tbody/tr[{i}]/td[5]/a').get_property('title')
    
    try:
        vdm = table.find_element(By.XPATH, f'//*[@id="yw1"]/table/tbody/tr[{i}]/td[6]/a').text
    except:
        vdm = 'nd'
    return dict(shirt_number=numero_maglia,id=id,name=player, role=role,birth_date = data_nasc, nationality= nat, market_value= vdm, team = team_name)


def listaTrasferimenti(box, is_acquisto: bool, list_transfers: list, team_name: str) -> None:
    try:
        trs= box.find_elements(By.TAG_NAME, 'tr')
        for  i in range(1, len(trs)):
            tds = trs[i].find_elements(By.TAG_NAME, 'td')
            player = tds[0].find_element(By.TAG_NAME, 'a')
            player_name = player.text
            id = player.get_property('href').split('/')[-1]
            nat =tds[2].find_element(By.TAG_NAME,'img').get_property('title')

            try:
                second_team = tds[6].find_element(By.TAG_NAME, 'a').get_property('title')
            except:
                second_team = 'ND'
            cost = tds[8].text
            team_buyer = team_name if is_acquisto else second_team
            team_vendor = second_team if is_acquisto else team_name
            transf = dict(player_name=player_name,player_id=id,nationality= nat, team_buyer = team_buyer, team_seller =team_vendor,cost= cost)
            list_transfers.append(transf)
    except NoSuchElementException:
        print(f'{team_name} nessuna operazione, is_acquisto {is_acquisto}')
        
def initializeDriver():
    driver = webdriver.Chrome()
    driver.get('https://www.transfermarkt.it/')
    acceptCookie(driver)
    return driver


def mainRose(competitions, seasons):
    driver = webdriver.Chrome()
    driver.get('https://www.transfermarkt.it/')
    acceptCookie(driver)

    for league, url in competitions.items():
        for s in seasons:
            if league == 'laliga' and s == '2023':
                try:
                    print(f'Inizio {league}, url: {url+s}')
                    teams = getTeamsURLS(driver, url+s)
                    players = []
                    #driver.quit()
                    for team in teams:
                        #print(team)
                        driver.get(team)
                        team_name = getTeamName(driver)
                        n = NumberPlayers(driver)
                        #print(n)
                        for i in range(1, n+1):
                            players.append(getPlayersInfo(driver, i, team_name))
                    dataset_path = 'Dataset\Transfermarkt'
                    writeJson(players, os.path.join(dataset_path, f'{league}_{s}.json'))
                    print(f'Fine {league} {s}')
                except Exception as e:
                    print(e)
                #driver.quit()

    driver.quit()


    if __name__ == '__main__':
        out_path = 'Dataset\Transfermarkt\Transfers'
        file_path = 'Dataset//Transfermarkt//transfers_urls.json'
        driver = initializeDriver()
        with open(file_path, 'r') as file:
            competitions = json.load(file)
        season = ['2019','2020','2021','2022','2023']
        #season = ['2023']
        for league, url in competitions.items():
            if league in ['eredivisie']:
                for s in season:
                    url_n = url + f'/plus/?saison_id={s}&s_w=&leihe=3&intern=0'
                    print(url_n) 
                    driver.get(url_n)
                    boxes = driver.find_elements(By.CLASS_NAME, 'box')[3:-1]
                    list_transfers = []
                    for x in boxes:
                        team_name = x.find_element(By.CLASS_NAME, 'content-box-headline').text
                        transfers = x.find_elements(By.CLASS_NAME, 'responsive-table')
                        acquisti = transfers[0]
                        cessioni = transfers[1]
                        listaTrasferimenti(acquisti, True, list_transfers, team_name)
                        listaTrasferimenti(cessioni,False, list_transfers, team_name)
                    writeJson(list_transfers, os.path.join(out_path, f'{league}_{s}.json'))
        driver.quit()