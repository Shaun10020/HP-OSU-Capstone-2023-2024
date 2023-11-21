import scrapy
import csv
from selenium import webdriver 
from selenium.webdriver.common.by import By
import time
from datetime import datetime, timezone
from pathlib import Path
from tldextract import extract
 
class PdfSpider(scrapy.Spider):
    ## Name for spider
    name = "pdf_spider"

    def start_requests(self):
        
        ## Changing websites one at a time
        domains = ["fema.gov"]
        
        ## Open browser
        browser = webdriver.Chrome() 
        
        ## Set true if google requires CAPTCHA test
        check = False
        
        ## Loop each domain if has move than doamin
        for domain in domains:
            ## Google search query
            search_string = "site:"+domain+" filetype:pdf"
            urls = []
            
            ## Check up to 40 pages in google search
            for i in range(40): 
                browser.get("https://www.google.com/search?q=" + search_string + "&start=" + str(i*10))
                ## 40 seconds to manually do CAPTCHA test 
                if (i == 0 and check):
                    time.sleep(40)
                ## Find google search result links
                elem = browser.find_elements(By.XPATH,'//div[@class="MjjYud"]/div/div/div/div/div/span/a')
                url = [i.get_attribute("href") for i in elem]
                ## Append to urls list
                urls = urls + url
        
            folder = Path(domain)
            folder.mkdir(parents=True, exist_ok=True)
            ## Create csv file to store filename, url, datetime
            filepath = folder/"source.csv"
            with open(filepath, 'w', newline='') as csvfile:
                pdfwriter = csv.writer(csvfile, delimiter=',',)
                pdfwriter.writerow(['Filename','URL','Download Date/Time (UTC format)'])
            ## parse pdf for each url
            for url in urls:
                yield scrapy.Request(url=url, callback=self.parse)
                
        ## Close browser
        browser.close()
            
    ## Parse pdf files
    def parse(self, response):
        ## Get domain, subdomain, pdf filename
        url = extract(response.url)
        folder = url.domain +"." + url.suffix
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        pdf = response.url.split('/')[-1]
        
        ## Parse robots.txt
        yield scrapy.Request(url="https://"+url.subdomain +"."+url.domain +"."+ url.suffix+"/robots.txt",callback=self.parse_robots)
        
        ## Get time and save pdf
        self.logger.info('Saving PDF %s', pdf)
        time = datetime.now(timezone.utc)
        pdf = url.subdomain+"_"+pdf
        path = folder/pdf
        csvfile = folder/"source.csv"
        with open(path, 'wb') as f:
            f.write(response.body)
        with open(csvfile, 'a', newline='') as csvfile_writer:
            pdfwriter = csv.writer(csvfile_writer, delimiter=',',)
            pdfwriter.writerow([pdf,response.url,time])
            
    ## Parse robots text file
    def parse_robots(self,response):
        url = extract(response.url)
        folder = url.domain +"." + url.suffix
        folder = Path(folder)
        filename = url.subdomain +"."+url.domain +"."+ url.suffix+"_robots.txt"
        path = folder/filename
        with open(path, 'wb') as f:
            f.write(response.body)
            

