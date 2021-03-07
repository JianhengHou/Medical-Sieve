from . import config
import shutil 
import scrapy
from patient.items import PatientItem
import re
import json
import os


class PatientDiscussionSpider(scrapy.Spider):
    # Make sure GROUP_PROFILING file and DATASET directory don't exsit, otherwise to move and create new.
    try:
        os.remove(config.GROUP_PROFILING_PATH)
        open(config.GROUP_PROFILING_PATH, 'w')
    except:
        open(config.GROUP_PROFILING_PATH, 'w')
    try:
        shutil.rmtree(config.RAW_DATASET_PATH)
        os.mkdir(config.RAW_DATASET_PATH) 
    except:
        os.mkdir(config.RAW_DATASET_PATH) 

    # Specify scrapy prroject name and domains alloweds
    name = config.SPIDERNAME
    allowed_domains = [config.TARGET_DOMAIN]

    def __init__(self):
        self.file_dic = {}
        self.numOfExpectedPages_dic = {}
        self.page_count = 0

    def toDigits(self, string):
        result = ""
        for char in string:
            if char >= '0' and char <= '9':
                result += char
        return int(result)
    
    def start_requests(self):
        # to build group profile for each group and crawl data by groups in the category list
        comunity_page = config.CONMUNITY_FORUM_PAGE_URLS
        for url in comunity_page:
            yield scrapy.Request(url=url, callback=self.traversal_category_pages)

        # # to crawl discussion post for each group by alphabetical order [discarded]
        # start_urls = []
        # seed = "https://patient.info/forums/index-"
        # for number in range(97, 123):
        #     start_urls.append(seed+chr(number))
        # for url in start_urls:
        #     yield scrapy.Request(url=url, callback=self.traversal_group_pages)

    def traversal_category_pages(self, response):
        category_links = response.css("ul[class='con-meds-lst grid-container no-gutters list-unstyled'] li[class='con-meds-item col--medium-6'] a[class='con-meds-lnk']::attr(href)").extract()
        for category_link in category_links:
            url =  "https://" + config.TARGET_DOMAIN + "/" + category_link
            yield scrapy.Request(url=url, callback=self.build_profile)

    def build_profile(self, response):
        category = response.css("div[class='masthead'] h1[class='articleHeader__title masthead__title']::text").extract()[0]
        category = re.sub(r'[^\w\s-]', '', category)
        category_name = "_".join(category.replace("-", " ").split())
        
        # record stats of the group
        groups = response.css("ul[class='thread-list'] li[class='cardb'] div[class='cardb__block'] div[class='avt avt-sm']")
        
        for i in range(len(groups)):
            topic = groups[i].css( "div[class='avt-sm-cat-jl'] h3[class='title'] a::text").extract()[0]
            topic = re.sub(r'[^\w\s-]', '', topic)
            group_name = "_".join(topic.replace("-", " ").split())
            discussion_string = groups[i].css( "div[class='col--small-6 col--medium-3 u-db'] span[class='last-reply-on'] strong::text").extract()[0]
            reply_string = groups[i].css( "div[class='col--small-6 col--medium-3 u-hide u-db--medium'] span[class='last-reply-on'] strong::text").extract()[0]
            member_string = groups[i].css( "div[class='col--small-6 col--medium-3 u-hide u-db--medium'] span[class='last-reply-on'] strong::text").extract()[1]
            reply_count = self.toDigits(reply_string)
            discussion_count = self.toDigits(discussion_string)
            profile_dict = {'category': category_name, 
                            "group":group_name, 
                            "member": self.toDigits(member_string), 
                            "post": discussion_count+reply_count, 
                            "reply": reply_count, 
                            "discussion": discussion_count}
            self.stat_file = open(config.GROUP_PROFILING_PATH, 'a')
            self.stat_file.write(json.dumps(profile_dict) + '\n')
            self.stat_file.close()
            group_link =  groups[i].css( "div[class='avt-sm-cat-jl'] h3[class='title'] a::attr(href)").extract()[0]
            group_link =  "https://" + config.TARGET_DOMAIN + "/" + group_link
            yield scrapy.Request(url=group_link, callback=self.parse)
        return 

        
    def parse(self, response):
        topic = response.css("div[class='masthead masthead-group masthead__padd'] h1[class='articleHeader__title masthead__title']::text").extract()[0].strip()
        topic = re.sub(r'[^\w\s-]', '', topic)
        name = "_".join(topic.replace("-", " ").split())
        
        if not self.numOfExpectedPages_dic.__contains__(name):
            #  initialize dict to store maximum number of pages for each diseases
            self.numOfExpectedPages_dic[name] = {}
            max_page = response.css("select[name='page'][class='submit reply__control reply-pagination'] option ::text").extract()
            self.numOfExpectedPages_dic[name]["max_page"] = int(max_page[0].split('/')[-1]) if len(max_page) > 0 else 1
            self.numOfExpectedPages_dic[name]["current_page"] = 1
            # create the file to store corresponding diseases discusssion
            current_file = open(config.RAW_DATASET_PATH + "/" + name + '.jl','a')
            self.file_dic[name] = current_file
        else:
            self.numOfExpectedPages_dic[name]["current_page"] += 1
        #  check if it reachs the page at the end
        if self.numOfExpectedPages_dic[name]["current_page"] > self.numOfExpectedPages_dic[name]["max_page"]:
            return

        # follow links to other post pages
        for href in response.css("a[rel='discussion'][title='View replies']::attr(href)").extract():
            complet_href = "https://" + config.TARGET_DOMAIN + href
            yield scrapy.Request(complet_href, callback=self.disscusion_parse)
            
        # next discussion list redirect
        nextPageUrl = response.css("link[rel='next'] ::attr(href)").extract_first()
        if nextPageUrl != None:
            yield scrapy.Request(nextPageUrl, callback=self.parse)  
    
    def disscusion_parse(self, response): 
        if response.url.find(config.DISCUSSION_PAGE_URL_PREFIX) != -1:
            self.page_count += 1
            item = PatientItem()

            # extract url
            item['url'] = response.url

            # extact content id
            content_id = response.css("meta[property='og:url']::attr(content)").extract()[0]
            content_id = re.sub(r'^.+-([^-]+)$', r'\1', content_id)
            item['content_id'] = content_id 

            # extract category
            path = response.css("div[class='container articleHeader__tools'] ol[class='breadcrumbs'] li[class='breadcrumb-item'] a span::text").extract()
            category = path[-2]
            category = re.sub(r'[^\w\s-]', '', category)
            category_name = "_".join(category.replace("-", " ").split())
            item['category'] = category_name

            # extract group
            gp = path[-1]
            gp = re.sub(r'[^\w\s-]', '', gp)
            gp_name = "_".join(gp.replace("-", " ").split())
            item['group'] = gp_name

            # writer part
            writer = response.css("div[class='author'] h5[class='author__info'] a::text").extract()[0]
            post = response.xpath("//div[@id='post_content'][@class='post__content']/p//text()").extract()

            post_content =  " ".join(line.strip() for line in post[:-1])
            post_timestamp  = response.css("div[id='topic'][class='post__main'] p[class='post__stats'] time[class='fuzzy']::attr(datetime)").extract_first()
            dict_post = {'poster':writer, 'text':post_content, 'timestamp':post_timestamp}
            item['post'] = dict_post

            # replies part
            dict_reply = {}
            reply = response.css("ul[class='comments'] li[class='comment'] article[class='post post__root']")
            count = 0
            for i in range(len(reply)):
                responsers = reply[i].css("div[class='post__header'] h5[class='author__info'] a[rel='nofollow author']::text").extract_first()
                response_time = reply[i].css("div[class='post__header'] p[class='post__stats'] time[class='fuzzy']::attr(datetime)").extract_first()
                response_text_list = reply[i].xpath("div[@class='post__content break-word'][@itemprop='text']/p//text()").extract()
                response_text = " ".join(line.strip() for line in response_text_list)
                # original version: aggregate all writer's post together, now discarded
                # if responsers == writer:
                #     post_content.join(response_text)
                # else:
 
                if response_text != "":
                    count += 1
                    dict_reply[count] = {'poster':responsers, 'text':response_text, 'timestamp':response_time}

                nested_reply = reply[i].css("ul[class='comments comments--nested'] li[class='comment comment--nested'] article[class='post']")

                for m in range(len(nested_reply)):
                    nested_responsers = nested_reply[m].css("div[class='post__header'] h5[class='author__info'] a[rel='nofollow author']::text").extract_first()
                    nested_response_time = nested_reply[m].css("div[class='post__header'] p[class='post__stats'] time[class='fuzzy']::attr(datetime)").extract_first()
                    nested_response_text_list = nested_reply[m].xpath("div[@class='post__content break-word'][@itemprop='text']/p//text()").extract()
                    nested_response_text = " ".join(line.strip() for line in nested_response_text_list)
                    # original version: aggregate all writer's post together, now discarded
                    # if nested_responsers == writer:
                    #     post_content.join(nested_response_text)
                    # else:
                    if nested_response_text != "": 
                        count += 1
                        dict_reply[count] = {'poster':nested_responsers, 
                                             'text':nested_response_text, 
                                             'timestamp':nested_response_time}
            item['reply'] = dict_reply

            # next reply page
            nextPageUrl = response.css("a[class='reply__control reply-ctrl-last link']::attr(href)").extract_first()
            if nextPageUrl != None:
                yield scrapy.Request(nextPageUrl, callback=self.disscusion_parse)
            
            item['group'] = re.sub(r'[^\w\s-]', '', item['group'])
            name = "_".join(item['group'].replace("-", " ").split())
            
            self.file_dic[name].write(json.dumps(dict(item))+ '\n')

 
