#!/usr/bin/env python

import re
import urllib.request

def get_content(url):
    with urllib.request.urlopen(url) as response:
        return response.read()

def get_html(url):
    return str(get_content(url))

entry_link = re.compile(re.escape('href="http') +
                        's?' +
                        re.escape('://zxart.ee/') +
                        '([^"]*)' +
                        re.escape('" class="pictures_list_image_link"'))

source_link = re.compile(re.escape('http') +
                        's?' +
                         re.escape('://zxart.ee/image/type:inspiredImage/') +
                         '([^\'\"]*)')

scr_link = re.compile(re.escape('http') +
                        's?' +
                      re.escape('://zxart.ee/file/id:') +
                      '[0-9]+/' +
                      '([^\'\"]*)')

count = 1

def find(regexp, html):
    m = regexp.search(html)
    if m is None:
        return None
    t = m.group(0)
    if t.endswith('\\'):
        t = t[:-1]
    return t

def download(url, name):
    data = get_content(url)
    with open(name, "wb") as f:
        f.write(data)

def process_entry(entry):
    global count
    html = get_html(entry)
    source_file = find(source_link, html)
    scr_file = find(scr_link, html)
    if source_file is None or scr_file is None:
        print("Cannot parse", entry)
        return
    if not scr_file.endswith('.scr'):
        print('Not know about', scr_file)
        return
    print(source_file)
    download(source_file, 'pairs/{}'.format(count))
    print(scr_file)
    download(scr_file, 'pairs/{}.scr'.format(count))
    count += 1

def process_query_page(html):
    entries = entry_link.findall(html)
    for entry in entries:
        process_entry('http://zxart.ee/' + entry)
    return len(entries)

def fetch_query(query_template):
    page = 1
    while True:
        print("Page", page)
        url = query_template if page == 1 else query_template + 'page:{}/'.format(page)
        html = get_html(url)
        if 0 == process_query_page(html):
            break
        page += 1

main_query = 'http://zxart.ee/eng/graphics/database/sortParameter:votes/sortOrder:desc/inspiration:1/resultsType:zxitem/'

if __name__ == "__main__":
    fetch_query(main_query)
    print(count)
