import scrapy

class autoCrawl(scrapy.Spider):
    name = "autoCrawl"
    start_urls = ['http://www.wisepoll.com/index.php/Sentence/lists/kid/653/mfid/0/ffid/0/status/-1/sort/1/feature_word_id/0/p/1']
    autoID = 29
    count = 1
    ur_part1 = 'http://www.wisepoll.com/index.php/Sentence/lists/kid/'
    ur_part2 = '/mfid/0/ffid/0/status/-1/sort/1/feature_word_id/0/p/'
    fi = open('../../data.txt',"a",encoding="utf-8")
    
    def parse(self,response):
        commentList = response.css('.c-l-comments')
        for val in commentList:
            comment = arrayToString(val.css('.article_a::text').extract())
            if comment.strip():
                self.fi.write(comment+'\n')
                print(comment)
        if commentList and self.count<100:
            self.count = self.count+1
            yield scrapy.Request(self.ur_part1+str(self.autoID)+self.ur_part2+str(self.count),callback=self.parse)
        elif self.autoID<1000:
            self.autoID = self.autoID+1
            self.count = 1
            yield scrapy.Request(self.ur_part1+str(self.autoID)+self.ur_part2+str(self.count),callback=self.parse)
        else:
            self.fi.close()

def arrayToString(arr):
    result = ''
    for val in arr:
        result += val
    return result