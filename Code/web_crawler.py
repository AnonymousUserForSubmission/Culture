import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import os

# If you don't have it, install: pip install googlesearch-python
# or: pip install beautifulsoup4
try:
    from googlesearch import search, get_response
except ImportError:
    raise ImportError("Please install 'googlesearch-python' with `pip install googlesearch-python`.")

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/90.0.4430.93 Safari/537.36'
}


def search_and_crawl(query, num_results=3, pause_seconds=2):
    """
    Perform a Google search for the given query, retrieve top URLs, and crawl their content.

    Args:
        query (str): The keyword or query to search.
        num_results (int): Number of top results to fetch.
        pause_seconds (int): Delay between search requests (to respect Google's rate limits).

    Returns:
        dict: Mapping of URL -> raw HTML page content (string).
    """
    crawled = {}
    print(f"Searching for '{query}' and fetching top {num_results} results...")
    try:
        for url in search(query, num_results=num_results, sleep_interval=pause_seconds, lang='en'):
            print(f"Crawling {url}")
            html = crawl_url(url)
            crawled[url] = html
    except Exception as e:
        print(f"Error during search/crawl: {e}")
    return crawled


def crawl_url(url, timeout=10):
    """
    Fetch the raw HTML content of a URL.

    Args:
        url (str): Webpage URL.
        timeout (int): Timeout for the HTTP request in seconds.

    Returns:
        str: Raw HTML content of the page, or an empty string on failure.
    """
    try:
        # response = requests.get(url, headers=HEADERS, timeout=timeout)
        # response.raise_for_status()
        session = requests.Session()
        session.headers.update(HEADERS)
        response = session.get(url)
        return response.text
    except requests.RequestException as e:
        print(f"Failed to fetch {url}: {e}")
        return ""


def extract_text(html):
    """
    Extract and clean visible text from raw HTML.

    Args:
        html (str): Raw HTML.

    Returns:
        str: Cleaned text extracted from HTML.
    """
    soup = BeautifulSoup(html, 'html.parser')
    # Remove scripts and styles
    for tag in soup(['script', 'style']):
        tag.decompose()
    text = soup.get_text(separator=' ', strip=True)
    return text


def search_and_try_summary(query, link_xpath, text_xpath):
    print("Searching for summary of '{query}'...".format(query=query))
    response = get_response(query)
    if response.status_code == 200:
        html = response.text
    else:
        print("Failed to fetch {query}: {response.status_code}")
        return None, None

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(options=chrome_options)
    # driver.get("data:text/html;charset=utf-8," + html)
    local_html_path = os.path.abspath("google.html")
    driver.get(f"file:///{local_html_path}")

    link_element, text_element = None, None
    for x_path in link_xpath:
        try:
            link_element = driver.find_element(By.XPATH, x_path)
        except NoSuchElementException:
            continue
        if link_element:
            break

    for x_path in text_xpath:
        try:
            text_element = driver.find_element(By.XPATH, x_path)
        except NoSuchElementException:
            continue
        if text_element:
            break

    snippet_container = text_element.text.strip() if text_element else None
    href = link_element.get_attribute('href') if text_element else None

    if snippet_container:
        featured_text = snippet_container
        if href:
            link = href.split("&")[0].replace("file:///D:/url?q=", "")
        else:
            link = ""
        print(f"Found summary {featured_text} on {link}")
        return featured_text, link

    else:
        print("No featured snippet found. Now we try top3 pages.")
        return None, None


