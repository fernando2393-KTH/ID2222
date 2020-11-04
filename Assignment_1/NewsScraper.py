from bs4 import BeautifulSoup
import requests
import json

# Source to access the list of articles
SOURCE = 'https://english.elpais.com/news/spanish_news/'


def scrapeArticle(url, headers):
    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.content, 'html.parser')
    article = soup.find("article")
    raw_text = [i.text for i in article.find_all(["h1", "h2", "p"], recursive=True)]
    text = ' '.join(raw_text)
    return text


def main():
    # Define keys to access the list of articles
    primary_key = 'col desktop_8 tablet_8 mobile_4'
    secondary_key = 'col desktop_8 tablet_6 mobile_4'

    # Define the system to access
    headers = requests.utils.default_headers()
    headers.update({
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0',
    })

    # Requests
    page = requests.get(SOURCE, headers=headers)
    soup = BeautifulSoup(page.content, 'html.parser')
    articles_column = soup.find("div", class_=primary_key)
    articles = articles_column.find_all("article")

    for idx, article in enumerate(articles):
        print('Article ', idx + 1)
        container = article.find("div", class_=secondary_key)
        link = container.find("a", href=True)
        url = 'https://english.elpais.com/' + link['href']
        # Read articles
        text = scrapeArticle(url, headers)
        # Save articles
        info = {'id': idx,
                'text': text
                }
        with open('data/' + str(idx) + 'article.json', 'w', encoding='utf8') as json_file:
            json.dump(info, json_file, ensure_ascii=False)


if __name__ == "__main__":
    main()
