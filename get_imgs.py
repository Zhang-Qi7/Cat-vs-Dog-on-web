import os
import re
import requests

def download_baidu(word):

    url = 'https://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word=' + word + '&ct=201326592&v=flip'
    pic_url = re.findall('"objURL":"(.*?)",', requests.get(url).text, re.S)

    i = 0
    for each in pic_url:
        print(pic_url)
        try:
            pic = requests.get(each, timeout=10)
        except requests.exceptions.ConnectionError:
            print('exception')
            continue

        if word in ['dog','dogs','puppy','狗']:
            word = 'dog'
        if word in ['cat','cats','猫']:
            word = 'cat'
            
        string = './train/' + word + '/' + word + '_' + str(i) + '.jpg'
        fp = open(string, 'wb')
        fp.write(pic.content)
        fp.close()
        i += 1

if __name__ == '__main__':

    word = input("Input key word: ")
    download_baidu(word)
