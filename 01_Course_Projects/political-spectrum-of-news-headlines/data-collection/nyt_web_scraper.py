import requests
from bs4 import BeautifulSoup
import re
import os
import langdetect
#create base url and cases
num_headlines=0
error_count=0
base_url = "https://www.nytimes.com/sitemap/"
url = "https://www.nytimes.com/sitemap/"
page = requests.get(url)
site=[]
author_names=[]
headline=[]
headline_date=[]
year_hrefs=[]
# find page content
soup = BeautifulSoup(page.content, "html.parser")
# get all year urls
year_block=soup.find("ol", class_="css-7ybqih")
year_links=year_block.find_all("a")

for link in year_links:
    href=link.get("href")
    year_hrefs.append(str(href))

year_hrefs=year_hrefs[0:10]
#loop through years to collect data
for ytext in year_hrefs:
    month_hrefs=[]
    #generate year_url
    year_url=base_url+str(ytext)
    print(year_url)
    year_page = requests.get(year_url)
    # get all month urls per year
    year_soup = BeautifulSoup(year_page.content, "html.parser")
    # print(ytext)
    # print(year_soup)
    try:
        # print("test year")
        month_block=year_soup.find("ol", class_="css-5emfqe")
        # print(month_block)
        month_links=month_block.find_all("a")
    except AttributeError:
        print("AttributeError")
        error_count+=1
        continue
    except KeyboardInterrupt:
        print(error_count)
        print(num_headlines)
        break
    else:

        for link in month_links:
            href=link.get("href")
            month_hrefs.append(str(href))
        #loop through months to collect data
        for mtext in month_hrefs:
            day_hrefs=[]
            #generate month_url
            month_url=year_url+str(mtext)
            month_page = requests.get(month_url)
            # get all month urls per year
            month_soup = BeautifulSoup(month_page.content, "html.parser")
            try:
                # get all day urls
                # print("test month")
                day_block=month_soup.find("ol", class_="css-7ybqih")
                day_links=day_block.find_all("a")
            except AttributeError:
                print("AttributeError")
                error_count+=1
                continue
            except KeyboardInterrupt:
                print(error_count)
                print(num_headlines)
                break
            else:
                for link in day_links:
                    href=link.get("href")
                    day_hrefs.append(str(href))
                #loop through days to collect data

                for dtext in day_hrefs:
                    #generate day_url
                    day_url=month_url+str(dtext)
                    day_page = requests.get(day_url)
                    headline_soup = BeautifulSoup(day_page.content, "html.parser")
                    # print(headline_soup)
                    try:
                        # print("test day")
                        headline_block=headline_soup.find("ul", class_="css-cmbicj")
                    # print(headline_block)
                        indv_headlines=headline_block.find_all("a")
                    except AttributeError:
                        print("AttributeError")
                        error_count+=1
                        continue
                    except KeyboardInterrupt:
                        print(error_count)
                        print(num_headlines)
                        break
                    else:
                        for item in indv_headlines:
                            print(f'{mtext+dtext+ytext},{str(item.string)}')
                            try:
                                if langdetect.detect(str(item.string))!="en":
                                    print("skipped")
                                    continue
                            except langdetect.lang_detect_exception.LangDetectException:
                                print("skipped")
                                continue
                            else:
                                headline.append(str(item.string))
                                headline_date.append(mtext+dtext+ytext)
                                num_headlines+=1
#write data into csv
dir_path = os.path.dirname(os.path.realpath(__file__))
headline_data=open(dir_path+'/data/nyt_headlines.csv', 'a')
headline_data.write("NYT\n")
for index in range(len(headline)):
    headline_data.write(headline_date[index])
    headline_data.write(","+ headline[index]+"\n")

headline_data.close()
print(error_count)
