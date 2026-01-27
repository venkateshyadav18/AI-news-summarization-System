import requests
from bs4 import BeautifulSoup

class NewsFetcher:
    def fetch_article(self, url: str) -> str:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return ""

            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all(["p"])
            text = " ".join(p.get_text() for p in paragraphs)
            return text.strip()
        except Exception as e:
            print(f"Error fetching article: {e}")
            return ""
