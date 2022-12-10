from bs4 import BeautifulSoup
import requests
import langdetect
import os
month = {	'01':'Janauary',
		'02':'February',
		'03':'March',
		'04':'April',
		'05':'May',
		'06':'June',
		'07':'July',
		'08':'August',
		'09':'September',
		'10':'October',
		'11':'November',
		'12':'December'		}
error_count=0
base_urls = ["https://www.foxnews.com/sitemap.xml",
            "https://www.bbc.com/sitemaps/https-index-com-archive.xml",
            "https://www.washingtonpost.com/wp-stat/sitemaps/index.xml",
            "https://www.csmonitor.com/sitemap-index.xml",
            "https://nypost.com/robots.txt",
            "https://www.breitbart.com/sitemap_news.xml",
            "https://www.motherjones.com/sitemap-index-1.xml",
            " https://www.washingtontimes.com/sitemap-stories.xml",
            "https://www.bbc.com/sitemaps/https-index-com-archive.xml"]
site=[]
author_names=[]
headlines=[]
headline_date=[]
year_hrefs=[]

def get_urls(url: str,parse_method="xml",id="loc"):
    urls=[]
    if parse_method=='txt':
        r = requests.get(url)
        file = r.text
        for line in file.split("\n"):
            if line.startswith(id):
                urls.append(line.split(": ")[1])
        return urls
    else:
        r = requests.get(url)
        file = r.text
        soup = BeautifulSoup(file,parse_method)
        if id:
            urls = soup.find_all(id)
        return urls,soup

def is_eng(line):
    try:
        return langdetect.detect(line)=="en"
    except langdetect.lang_detect_exception.LangDetectException:
        return False

def to_number_date(date):
    a=date.split(" ")
    a[0]=list(month.keys())[list(month.values()).index(a[0])]
    a[1]=a[1].replace(',','')
    if len(a[1])==1:
        a[1]="0"+a[1]
    return "-".join(a)

    

for i in range(len(base_urls)):
    i=3
    headlines=[]
    url=base_urls[i]
    print(f"{url.split('/')[2]}\n")
    if i==0:
        sitemap_urls,sitemap_soup = get_urls(url)
        for submap in sitemap_urls:
            submap=submap.string.replace('amp;', '')
            if "type=articles&" not in submap:
                    continue
            article_urls,article_soup = get_urls(submap)
            article_dates=article_soup.find_all("lastmod")
            length=0
            for index in range(len(article_urls)):
                article_url=article_urls[index]
                date=article_dates[index]
                print(article_url.string)
                headline=article_url.string.replace(url.split('.com/')[1],'').split('/')
                for h in headline:
                    if "-" in h:
                        headline=h
                        break
                headline=" ".join(headline.split('-'))
                print(headline)
                if is_eng(headline):
                    headlines.append(headline)
                    headline_date.append(date.string[0:10])
                    print(f"{date.string[0:10]},{headline}")
                    length+=1
                    if length==100:
                        length=0
                        break
                else:
                    print("skipped")
    elif i==1:
        length=0
        sitemap_urls,sitemap_soup = get_urls(url)
        for submap in sitemap_urls:
            article_urls,article_soup = get_urls(submap.string)
            article_dates=article_soup.find_all("lastmod")
            for index in range(len(article_urls)):
                article_url=article_urls[index]
                if article_url.string.split('.com/')[1].split('/')[0]!="news":
                    print('skipped')
                    continue
                url,article_soup=get_urls(article_url.string,"html",id=None)
                date=article_dates[index]
                print(article_url.string)
                if not article_soup.find("h1"):
                    print("skipped")
                    continue
                if article_soup.find("h1"):
                    headline=article_soup.find("h1").string
                else:
                    print("skipped")
                    continue
                print(headline)
                if headline:
                    if is_eng(headline):
                        headlines.append(headline)
                        headline_date.append(date.string[0:10])
                        print(f"{date.string[0:10]},{headline}")
                        length+=1
                    else:
                        print("skipped")
                if length%150==0:
                    break
            if length>=15000:
                break
    elif i==2:
        sitemap_urls,sitemap_soup = get_urls(url)
        for submap in sitemap_urls:
            article_urls,article_soup = get_urls(submap.string)
            article_dates=article_soup.find_all("lastmod")
            length=0
            for index in range(len(article_urls)):
                article_url=article_urls[index]
                date=article_dates[index]
                print(article_url.string)
                headline=article_url.string.split('.com/')[1].split('/')
                for h in headline:
                    if "-" in h:
                        headline=h
                        break
                headline=" ".join(headline.split('-'))
                print(headline)
                if is_eng(headline):
                    headlines.append(headline)
                    headline_date.append(date.string[0:10])
                    print(f"{date.string[0:10]},{headline}")
                    length+=1
                    if length==100:
                        length=0
                        break
                else:
                    print("skipped")
    elif i==3:
        sitemap_urls,sitemap_soup = get_urls(url)
        for j in range(len(sitemap_urls)):
            sitemap=sitemap_urls[j].string
            print(sitemap)
            if "-auto-1" not in sitemap:
                    print('skipped')
                    continue
            submap_urls,soup = get_urls(sitemap)
            for index in range(len(submap_urls)-3):
                article_url=submap_urls[index].string
                if get_urls(article_url.string,'html',None)[1].find("time"):
                    date=get_urls(article_url.string,'html',None)[1].find("time").string
                else:
                    print("skipped")
                    continue
                print(article_url.string)
                headline=article_url.string.split('/')[-1]
                print(headline)
                print(".html" in headline)
                if ".html" not in headline:
                    headline=" ".join(headline.split('-'))
                else:
                    headline=get_urls(article_url.string,'html',None)[1].find("h1").string.splitlines()[2].replace("/t",'')
                print(headline)
                if is_eng(headline):
                    headlines.append(headline)
                    headline_date.append(to_number_date(date).splitlines()[0])
                    print(f"{to_number_date(date).splitlines()[0]},{headline}")
                else:
                    print("skipped")
    elif i==4:
       sitemap_urls= get_urls(url,'txt','Sitemap:')
       for j in range(13,len(sitemap_urls)-6):
            sitemap=sitemap_urls[j]
            submap_urls,submap_soup = get_urls(sitemap)
            # print(sitemap)
            # print(submap_urls)
            for submap_url in submap_urls:
                print(submap_url.string)
                article_urls,article_soup = get_urls(submap_url.string)
                # print(article_urls)
                article_dates=article_soup.find_all("lastmod")
                # print(submap_url.string)
                # print(article_soup)
                dates_index=0
                for index in range(len(article_urls)):
                    if dates_index>len(article_dates)-1:
                        break
                    # print(len(article_urls))
                    # print(dates_index)
                    article_url=article_urls[index]
                    if "/wp-content/" in article_url.string:
                        continue
                    date=article_dates[dates_index]
                    print(article_url.string)
                    headline=article_url.string.replace(url.split('.com/')[1],'').split('/')
                    for h in headline:
                        if "-" in h:
                            headline=h
                            break
                    if isinstance(headline,list):
                        headline=headline[-2]
                    print(headline)
                    headline=" ".join(headline.split('-'))
                    if is_eng(headline):
                        headlines.append(headline)
                        headline_date.append(date.string[0:10])
                        print(f"{date.string[0:10]},{headline}")
                        dates_index+=1
                    else:
                        print("skipped")            
    elif i==5:
        sitemap_urls = get_urls(url)
    elif i==6:
       sitemap_urls = get_urls(url)
    elif i==7:
        sitemap_urls = get_urls(url)
    elif i==8:
        sitemap_urls = get_urls(url)
    elif i==9:
        sitemap_urls = get_urls(url)
        # if url=="https://www.washingtonpost.com/wp-stat/sitemaps/index.xml":
        #     for index in range(len(submap_urls)):
        #         print(index)
        #         submap=submap_urls[index]
        #         date=submap_dates[index]
        #         print(submap.string)
        #         headline=submap.string.replace('http://www.washingtonpost.com/archive/','').split('/')[4]
        #         try:
        #             if langdetect.detect(headline)!="en":
        #                 print("skipped")
        #                 continue
        #         except langdetect.lang_detect_exception.LangDetectException:
        #             print("skipped")
        #             continue
        #         else:
        #             headlines.append(" ".join(headline.split('-')))
        #             headline_date.append(date.string[0:9])
        #             print(f"{date.string[0:9]},{headline}")
        #             length+=1
        #             if length==100:
        #                 length=0
        #                 break
    #write data file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    headline_data=open(dir_path+f"/data/{url.split('.')[1]}_headlines.csv", 'w')
    headline_data.write(f"{url.split('.')[1]}\n")
    for index in range(len(headlines)):
        headline_data.write(headline_date[index]+", ")
        headline_data.write(headlines[index]+"\n")
    headline_data.close()
    print(f"done {url}")
    break
