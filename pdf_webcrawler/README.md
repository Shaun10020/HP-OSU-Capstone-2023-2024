### Introduction
The web scrapper script is created using scrapy. Scrapy is an open-source web-crawling framework written in Python. 

### Requirement
- Python >= 3.11.5
- Scrapy >= 2.8.0
- Selenium >= 4.16.0
- tldextract >= 3.2.0

### How to use it 
##### Domains/Websites to scrape
In the script `pdf_spider.py`, modify the variable `domains` in the function `start_requests` to put domains in the array. For example, `domains = ["fema.gov"]` to scrape `fema.gov` website.

##### Manual bypass CAPTCHA
In the function `start_requests` from script `pdf_spider.py`, there is another variable called `check`, when it is set to `False`, the script will start scraping without a pause and start scraping pdfs. When it is set to `True`, the script will first pause for 40 seconds, to allow a window to do CAPTCHA test.

##### How to run it
Open a terminal and change the to `pdf_webcrawler` folder, for example, change the directories in terminal to 
`C:/YOUR_DIRECTORIES/pdf_webcrawler`.
and then run 
`scrapy crawl pdf_spider`
in the terminal.
