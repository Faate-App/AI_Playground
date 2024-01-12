from playwright.sync_api import sync_playwright
import re
import pandas as pd
import time
from bs4 import BeautifulSoup

def main():
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=False)

            
            page = browser.new_page()
            page.goto("https://www.linkedin.com/uas/login?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fsearch%2Fresults%2Fall%2F%3FfetchDeterministicClustersOnly%3Dtrue%26heroEntityKey%3Durn%253Ali%253Aorganization%253A1586%26keywords%3Damazon%26origin%3DRICH_QUERY_TYPEAHEAD_HISTORY%26position%3D0%26searchId%3Dafc6a59f-24b2-4236-8152-8fa35a3fe166%26sid%3DVPh&fromSignIn=true&trk=cold_join_sign_in", wait_until='networkidle')
           
            email = "henri.serano@gmail.com"
            mdp = "Utichang1"
            prenom = "Bill"
            nom = "Gates"
            
            email_field = page.wait_for_selector("id=username")
            email_field = page.query_selector("id=username")
            email_field.fill(email)
            password_field = page.wait_for_selector("id=password")
            password_field = page.query_selector("id=password")   
            password_field.fill(mdp)
            password_field.press("Enter")
            try:
                    page.goto("https://www.linkedin.com/search/results/people/?keywords="+str(prenom)+" "+str(nom)+"&origin=SWITCH_SEARCH_VERTICAL&sid=mqt", wait_until='networkidle')
                    
                    profil_link_selector = "a.app-aware-link[href*='linkedin.com/in']"
                    profil_link_element = page.query_selector(profil_link_selector)
                    
                    if profil_link_element:
                        profil_href = profil_link_element.get_attribute("href")
                        print(f"Le lien du profil est : {profil_href}")
                        # Naviguer vers le profil
                        page.goto(profil_href, wait_until='networkidle')
                    #page.goto("https://www.linkedin.com/in/williamhgates/", wait_until='networkidle')
                        
                    try:
                        
                        localisation = page.inner_text("span.text-body-small.inline.t-black--light.break-words")
                        print(f"Localisation: {localisation}")
                    except:
                        print("Problem with "+str(prenom)+" "+str(nom))
                    try:
                        titre = page.inner_text("div.JLTOxdAyPIbwAVfujRhErCSfXZcrNbXpHcBIE")
                        print(f"Titre: {titre}")
                    except:
                        print("Problem with "+str(prenom)+" "+str(nom))
                    try:
                        education = page.inner_text("section.pvs-list")   
                        print(education)
                        print(f"Education: {education}")
                    except:
                        print("Problem with "+str(prenom)+" "+str(nom))
                        

            except:
                    print("Problem with "+str(prenom)+" "+str(nom))
                    pass
                

                
            
        except Exception as e:
                print(f"An error occurred while scraping: {e}")


if __name__ == "__main__":
    main()
    
