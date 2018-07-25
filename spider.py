import requests
from requests.exceptions import RequestException
import json
from bs4 import BeautifulSoup

def get_onepage(url):
    try:
        response=requests.get(url)
        if response.status_code==200:
            return response.text
        return None
    except RequestException:
        return None

def get_urllist(html):
    urllist=[]
    soup = BeautifulSoup(html, 'lxml')
    for elem in soup.find_all(class_='storytitle'):
        print(elem.a.attrs['href'])
        urllist.append(elem.a.attrs['href'])
    return urllist

def parse_onepage(url,Serial_number):
     html2= get_onepage(url)
     soup = BeautifulSoup(html2, 'lxml')
     title_node= soup.find(class_="contenttitle")
     title=title_node.a.string
     write_to_file(title, Serial_number)
     entry = soup.find(name="textarea")
     ps = entry.findAll(name="p")
     for p in ps:
         write_to_file(p.text, Serial_number)

def write_to_file(content,Serial_number):
    with open(str(Serial_number)+'科普文章.txt','a',encoding='utf-8') as f:
        f.write(json.dumps(content,ensure_ascii=False)+'\n')

def main(offset):
    url='http://songshuhui.net/archives/tag/%E5%8E%9F%E5%88%9B/page/'+str(offset)
    html_urllist=get_onepage(url)
    urllist=get_urllist(html_urllist)
    Serial_number=15*(offset-1)
    for url_item in urllist:
      Serial_number+=1
      parse_onepage(url_item,Serial_number)

if __name__=='__main__':
    Serial_number = 0
    for i in range(335,382):
      main(offset=i)













# import requests
# from bs4 import BeautifulSoup
# response=requests.get('http://songshuhui.net/archives/101591')
# # soup=BeautifulSoup(response.text,'lxml')
# # print(soup.find_all(class_='entry').find_all(name_='p'))
# # for item in soup.find_all(class_='entry'):
# #   print(item.content)
#
# soup = BeautifulSoup(response.text,'lxml')
# entry = soup.find(attrs={"class":"entry"})
# print("entry的值----------------------------------------------------")
# ps = entry.findAll("p")
# for p in ps:
#   #print("p的值----------------------------------------------------")
#   print(p.string)