#coding=utf-8
import requests
import json
import mimetypes
from PIL import ImageGrab
import datetime

def upload(files):
    APIKey = "YOUR API KEY"
    format = "json"
    #url = "http://domain.com/api/1/upload/?key="+ APIKey + "&format=" + format
    url = "https://sm.ms/api/upload?ssl=False&format=json"
    #files = 
    r = requests.post(url , files = files)

    return json.loads(r.text)

def formatSource(filename):
    imageList = []
    type = mimetypes.guess_type(filename)[0]
    imageList.append(('smfile' , (filename , open(filename , 'rb') , type)))
    print imageList
    return imageList

if __name__ == "__main__":
    print "将图片截图或复制到剪切板中即可~~，ctrl+z结束"
    recentVal = None
    while(True):
        tmpValue = ImageGrab.grabclipboard()
        if recentVal != tmpValue:
            recentVal = tmpValue
            now = datetime.datetime.now()
            now = now.strftime("%Y-%m-%d %H:%M:%S")
            filename = 'IMG'+ now + '.png'
            if recentVal is not None:
                recentVal.save(filename, 'png')
                #filenames.append(filename)
                #recentVal = None
                print filename
                jsonData = upload(formatSource(filename))

                if jsonData['code'] != "success":
                    print "error: " , jsonData['msg']
                    print "status code : " , jsonData['code']
                else:
                    print "url: " , jsonData['data']['url']
                    #print "orignal url: " , jsonData['image']['display_url']
                    #print "thumb url: " , jsonData['image']['thumb']['url']