def search_web(query):

    return [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://www.ibm.com/topics/artificial-intelligence",
        "https://www.britannica.com/technology/artificial-intelligence"
    ]

import requests
from bs4 import BeautifulSoup


def read_url(url):

    try:

        response = requests.get(
            url,
            timeout=10
        )

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        text = soup.get_text()

        return text[:2000]

    except Exception as e:

        return f"Error: {e}"
    
question = input(
    "Research Question: "
)
    
urls = search_web(question)

print("\nSources Found:\n")

for url in urls:

    print(url)

research_notes = []

for url in urls:

    content = read_url(url)

    research_notes.append(
        {
            "url": url,
            "content": content
        }
    )

print("\n=== RESEARCH NOTES ===\n")

for note in research_notes:

    print("\nURL:")
    print(note["url"])

    print("\nContent Preview:")
    print(note["content"][:500])

combined_text = ""

for note in research_notes:

    combined_text += (
        note["content"][:1000]
        + "\n\n"
    )

def generate_answer(question, text):

    summary = text[:1500]

    return f"""
Question:
{question}

Summary:
{summary}

Sources:
Research completed from 3 sources.
"""

def calculate_confidence(research_notes):

    valid_sources = 0

    for note in research_notes:

        content = note["content"]

        if (
            "Error" not in content
            and len(content) > 500
        ):
            valid_sources += 1

    if valid_sources >= 3:
        return "High (90%)"

    elif valid_sources == 2:
        return "Medium (70%)"

    else:
        return "Low (40%)"

answer = generate_answer(
    question,
    combined_text
)

print("\n=== FINAL ANSWER ===\n")

print(answer)

confidence = calculate_confidence(
    research_notes
)

print("\nConfidence:")

print(confidence)


