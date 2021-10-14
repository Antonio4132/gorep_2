#!/usr/bin/python3

import requests
import sys
import time

from lxml import html


def parse_name(name):
    name = ''.join(name).split(':')[0]
    name = " ".join(name.split())

    return name


def get_URI_from_name(term_name, ontology=""):
    seed_url = f"http://www.ontobee.org/search?ontology={ontology}&keywords={term_name}&submit=Search+terms"

    # User-Agent
    headers = {
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36"
    }

    answer = requests.get(seed_url, headers=headers)

    parser = html.fromstring(answer.text)

    terms = parser.xpath("//li[@class='search-list']")

    if not terms:
        return None

    seed_URI = terms[0].xpath("./a/text()")[0]

    return seed_URI


def get_name_from_URI(URI, ontology=""):
    if URI is None:
        return None

    seed_url = f"http://www.ontobee.org/search?ontology={ontology}&keywords={URI}&submit=Search+terms"

    # User-Agent
    headers = {
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36"
    }

    answer = requests.get(seed_url, headers=headers)

    time.sleep(2)

    parser = html.fromstring(answer.text)

    term = parser.xpath("//li[@class='search-list']")[0]

    name = term.xpath("./ul/li/text()")

    return parse_name(name)



def main(argv):
    if 2 < len(argv) > 3:
        print("Number of arguments incorrect.")
        print("USAGE: GetTermsOntobee <ontology>* <keyword>* <number_of_terms>")
        print("* meaning obligatory parameter.")
        sys.exit(1)

    ontology = argv[0]
    keyword = argv[1]
    n = float('inf')

    if len(argv) == 3:
        n = int(argv[2])

    seed_url = f"http://www.ontobee.org/search?ontology={ontology}&keywords={keyword}&submit=Search+terms"
    # User-Agent
    headers = {
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36"
    }

    answer = requests.get(seed_url, headers=headers)

    parser = html.fromstring(answer.text)

    terms = parser.xpath("//li[@class='search-list']")

    seed_URI = terms[0].xpath("./a/text()")[0]
    seed_ID = seed_URI.split('/')[-1]
    seed_url = f"http://www.ontobee.org/ontology/catalog/{ontology}?&iri=http://purl.obolibrary.org/obo/{seed_ID}&max=9999999"

    time.sleep(2)

    answer = requests.get(seed_url, headers=headers)

    parser = html.fromstring(answer.text)
    terms = parser.xpath("//li")

    n = min(n, len(terms))

    for term in terms[:n]:
        name = parse_name(term.xpath("./a/text()"))
        URI = term.xpath("./a/@href")[0].split('iri=')[-1]

        print(f"Name: {name}\nURI: {URI}")
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
