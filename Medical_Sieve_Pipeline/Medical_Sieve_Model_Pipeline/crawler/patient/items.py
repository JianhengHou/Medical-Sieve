# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

from scrapy.item import Item, Field


class PatientItem(Item):
    # define the fields for your item here like:
    content_id = Field()
    post = Field()
    reply = Field()
    url = Field()
    group = Field()
    category = Field()

    
