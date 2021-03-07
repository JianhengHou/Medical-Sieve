from pyspark import SparkContext, SparkConf
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
import re, unicodedata
import csv
import json
import os

class dataManager:
    def __init__(self, contraction_mapping, domain_term_mapping, fields):
        self.contraction_mapping = contraction_mapping
        self.domain_maping = domain_term_mapping 
        self.fields = fields
        
    def run_separator(self, input_dictionary_path, line_level_dataset_output_path, text_only_file_path):
        with open(line_level_dataset_output_path, 'w') as line_level_dataset_output, open(text_only_file_path, 'w') as text_only_file_output:
            csvWriter = csv.writer(line_level_dataset_output)
            csvWriter.writerow(self.fields)

            """use spark to process each file"""
            for filename in os.listdir(input_dictionary_path):
                if filename[-2:] != 'jl': continue
                conf = SparkConf() \
                        .setAppName("preprocessor") \
                        .set("spark.driver.memory", "32g")\
                        .set("spark.executor.memory", "16g")\
                        .set("spark.driver.host", "localhost")
                
                sc = SparkContext(conf=conf)
                inputData = sc.textFile(os.path.abspath(input_dictionary_path + '/' + filename))

                """break each post into discussion and replies into lines"""
                line_level_data = inputData.map(lambda x: json.loads(x))\
                                    .flatMap(lambda x: self.separate_discussion_and_reply((x['content_id'],(x['post'], x['reply'], x['group'], x['category']))))\
                                        .collect()        

                """write line in the list into a csv file"""
                for each in line_level_data:
                    csvWriter.writerow([each[field] for field in self.fields])
                    text_only_file_output.write(each['text_processed'] + '\n')
                sc.stop()

    def preprocess_terms(self, line):
        line = line.lower()

        """remove links"""
        line = re.sub(r"http\S+", "url ", line)
        line = re.sub(r"www.\S+", "url ", line)
        line = re.sub(r"([a-z]+)\.+([a-z]+)", r'\1 . \2', line)

        """replace some domain contraction words with their complete words"""
        for key, value in self.domain_maping.items():
            line = re.sub(key, value, line)

        """replace some general contraction words with their complete words"""
        word_list = []
        for word in line.split():
            new_word = word
            if self.contraction_mapping.__contains__(new_word):
                new_word = self.contraction_mapping[new_word]
            word_list.append(new_word)
        final_line = ' '.join(word_list).replace("'", "")
       
        return final_line

    def preprocess_punctuactions_and_links(self, line):
        """deal with punctuations"""
        line = re.sub(r'\?+', "? ", line)
        line = re.sub(r'\!+', "! ", line)
        line = re.sub(r'\.+\s', " . ", line)
        line = re.sub(r',[\w+\s]', " , ", line)
        line = line.replace('"',' " ') \
                   .replace(':',' : ')\
                   .replace('\n', ' ')\
                   .replace('/', ' ')\
                   .replace('_', ' ')\
                   .replace("\u00a0", " ")\
                   .replace("\u00a0", "")\
                   .replace("\u2019", "'")\
                   .replace("\u2018", "'")
        """fill period if there isn't one"""
        if "." not in line[-3:] and "?" not in line[-3:] and "!" not in line[-3:]:
            line = line + ' .'
        return line

    def normalize(self, word_pos):
        """
        input: tuple(word, pos)
        output: string
        """
        if len(word_pos) != 2: return ""
        """Remove non-ASCII characters from list of tokenized words"""
        new_word = unicodedata.normalize('NFKD', word_pos[0]).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_word = re.sub(r'[^\w\s\.\!\?\:\,\-\(\)}]', '', new_word)
        # lemmatize plural nouns only
        if word_pos[1] == 'NNP' or word_pos[1] == 'NNS': 
            lemmatizer = WordNetLemmatizer()
            new_word = lemmatizer.lemmatize(new_word, pos='n')
        return new_word

    def separate_discussion_and_reply(self, entry):
        result = []
        content_id = entry[0]
        category = entry[1][3]
        group = entry[1][2]

        # Discussion part
        sent_count = 0
        discussion_dict = entry[1][0]
        cleaned_raw_discussion = self.preprocess_punctuactions_and_links(discussion_dict['text'])
        for sent in sent_tokenize(cleaned_raw_discussion):
            sent_count += 1
            discussion_sent_entry = {}
            discussion_sent_entry["content_id"] = group + '_' + content_id + "-0-" + str(sent_count)
            discussion_sent_entry["post_type"] = "discussion"
            discussion_sent_entry["group"] = group
            discussion_sent_entry["category"] = category
            discussion_sent_entry["poster"] = discussion_dict["poster"]
            discussion_sent_entry["timestamp"] = discussion_dict["timestamp"]
            discussion_sent_entry["text"] = sent
            discussion_sent_entry["text_processed"] = " ".join([self.normalize(word) for word in pos_tag(word_tokenize(self.preprocess_terms(sent)))])
            result.append(discussion_sent_entry)

        # Reply part according to the discussion post above
        reply_count = 0
        for reply in entry[1][1].values():
            reply_count += 1
            sent_count = 0
            cleaned_raw_reply = self.preprocess_punctuactions_and_links(reply['text'])
            for sent in sent_tokenize(cleaned_raw_reply):
                sent_count += 1
                reply_sent_entry = {}
                reply_sent_entry["content_id"] = group + '_' + content_id + "-" + str(reply_count) + '-' + str(sent_count)
                reply_sent_entry["post_type"] = "reply"
                reply_sent_entry["group"] = group
                reply_sent_entry["category"] = category
                reply_sent_entry["poster"] = reply["poster"]
                reply_sent_entry["timestamp"] = reply["timestamp"]
                reply_sent_entry["text"] = sent
                reply_sent_entry["text_processed"] = " ".join([self.normalize(word) for word in pos_tag(word_tokenize(self.preprocess_terms(sent)))])
                result.append(reply_sent_entry)
        return result
   
    
