
import selenium
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from IPython.core.display import clear_output
import time

chrome_path = "/Users/mcullan/Documents/environments/my_env/bin/chromedriver"


#   Setup Headless Webdriver

# Get links from feed page
def setup_headless():
    options = Options()
    prefs = {"profile.managed_default_content_settings.images": 2}
    options.add_experimental_option("prefs", prefs)
    options.add_argument('--no-sandbox')
    options.add_argument('--window-size=1420,1080')
    options.headless = True
    options.add_argument('--disable-gpu')
    return options


# ## Get listing links for a given designer


def feed_items(designer='Commes Des Garcons', category = None,  n_pages = 5, headless = True, sold = True):
    # Initialize output variable.
    links = []
    
    # Start webdriver based on headless/not headless option
    if headless:
        driver = webdriver.Chrome(chrome_path, options = setup_headless())
    else:
        driver = webdriver.Chrome(chrome_path)
        
    if sold:
        driver.get('https://grailed.com/sold')
    else:
        driver.get('https://grailed.com')

    time.sleep(2)
    for a in range(10):
        driver.execute_script('window.scrollBy(0, 260)')
    
    time.sleep(1)
        
    print('went to grailed')
    # Open webdriver, enter designer, and scroll.
    try:
        # Navigate to element containing designer filter.
        print("finding designers wrapper")
        time.sleep(3)
        #driver.execute_script('window.scrollTo(0, 760)')

        try:
            wrapper = driver.find_element_by_class_name("designers-wrapper")
            print('got wrapper')
        except:
            print('failed getting wrapper')
        
        # Click element to expand filter window.
        wrapper.click()
        WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, "FiltersDesignerSearchForm")))
        
        # Click text box to type designer name.
        desform = driver.find_element_by_id('FiltersDesignerSearchForm')#.click()#.find_element_by_xpath('//label/input')
        desform.click()
        
        # Type designer name.
        inp = desform.find_element_by_class_name('search')
        inp.send_keys(designer)
        time.sleep(.25)
        clear_output(wait=True)
        
        # Click on button to filter by given designer.
        print("trying click")
        des = wrapper.find_element_by_class_name('indicator')
        des.click()
        time.sleep(1)
        clear_output(wait=True)
        print("clicked")
        
        if category:
            print('category')
            try:
                # Navigate to element containing category filter.
                cwrapper = driver.find_element_by_class_name('categories-wrapper')
                print('got wrapper')
                
                # Click element to expand filter window.
                cwrapper.click()
                
                # Click on button to filter by given designer.
                c = cwrapper.find_element_by_class_name(category+'-wrapper')
                print('found wrapper')
                time.sleep(2)
                m = c.find_element_by_class_name('indicator')
                print("found indicator")
                time.sleep(4)
                m.click()
                print("clicked "+ category)
                time.sleep(1)
                clear_output(wait=True)
            except:
                pass
        time.sleep(2)
        # Scroll the desired number of pages to display links.
        for i in range(n_pages):
            try: 
                for j in range(50):
                    driver.execute_script('window.scrollBy(0, 260)')
                    time.sleep(.01)
                time.sleep(.2)
                #driver.execute_script('window.scrollTo(0, document.body.scrollHeight)')

                clear_output(wait=True)
                print("Completed {} of {} scrolls".format(i, n_pages))
            except:
                pass
        driver.execute_script('window.scrollTo(0, document.body.scrollHeight)')

        #clear_output(wait=True)
        print("done scrolling")
        time.sleep(1)
       
        # Get all of the listing items on the page
        
        try:
            feed = driver.find_element_by_class_name('feed')
            feeditems = []
            working = True
            i=0
            for i in range( 80 * n_pages ):
                try:
                    feeditem = []
                    try:
                        feeditem = feed.find_element_by_xpath("""./div[""" + str(i + 1) + "]")
                        #print("found feed item")
                        info = {}
                        #print("trying to get image link")
                        info['text'] = feeditem.text
                        info['img'] = feeditem.find_element_by_tag_name('img').get_attribute('src')
                        info['category'] = category
                        info['designer'] = designer
                    except:
                        pass
                    if info['img']:
                        feeditems.append(info)
                except:
                   # print("no item for i = "+str(i))
                    pass
            print("Found {} feed items.".format(len(feeditems)))
        except:
            #clear_output(wait=True)
            print('Failed to get feed items.')
            driver.quit()
            pass

        
        
        driver.quit()
    except:
        print("failed")
        driver.quit()
        pass
   # clear_output()
    print('done!')
    return feeditems



def all_categories(designer='Commes Des Garcons', n_pages = 5, headless = True, sold = True):
    listings = []
    for category in ['tops', 'bottoms', 'outerwear','accessories']:
        try:
            listings.append(feed_items(designer = designer, category = category, headless = headless, sold = sold, n_pages = n_pages))
        except:
            pass            
    return listings


preme_unsold = all_categories(designer = "Supreme", n_pages = 50, headless = True, sold = False)
comme_sold = all_categories(designer="Comme Des Garcons", n_pages = 50, headless = True, sold = True)
comme_unsold = all_categories(designer = "Comme Des Garcons", n_pages = 50, headless = True, sold = False)
raf_sold = all_categories(designer="Raf Simons", n_pages = 50, headless = True, sold = True)
raf_unsold = all_categories(designer = "Raf Simons", n_pages = 50, headless = True, sold = False)
rick_sold = all_categories(designer="Rick Owens", n_pages = 50, headless = True, sold = True)
rick_unsold = all_categories(designer="Rick Owens", n_pages = 50, headless = True, sold = False)
balenciaga_sold = all_categories(designer="Balenciaga", n_pages = 50, headless = True, sold = True)
balenciaga_unsold = all_categories(designer="Balenciaga", n_pages = 50, headless = True, sold = False)
bape_unsold = all_categories(designer="Bape", n_pages = 50, headless = True, sold = False)
bape_sold = all_categories(designer="Bape", n_pages = 50, headless = True, sold = True)
gucci_sold = all_categories(designer="Gucci", n_pages = 50, headless = True, sold = True)
gucci_unsold = all_categories(designer="Guci", n_pages = 50, headless = True, sold = False)


unsold = comme_unsold + raf_unsold + rick_unsold + balenciaga_unsold+gucci_unsold+preme_unsold + yohji_unsold + undercover_unsold + ysl_unsold + helmut_unsold

sold = comme_sold + raf_sold + rick_sold + balenciaga_sold+gucci_sold+preme_sold + yohji_sold + undercover_sold + ysl_sold + helmut_sold

sold_flat = [[[listing for listing in category] for category in designer] for designer in sold]

unsold_done = []
for category in unsold:
    for item in category:
        unsold_done.append(item)
sold_done = []

for category in sold:
    for item in category:
        sold_done.append(item)            

dataset2 = {'unsold': unsold_done, 'sold': sold_done}

import json
with open('dataset.json', 'w') as outfile:
    json.dump(dataset, outfile)




