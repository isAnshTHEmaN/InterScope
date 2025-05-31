from openai import OpenAI
import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
import xmltodict
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def compute_gap_scores(keywordsOne, keywordsTwo, freqsOne, freqsTwo, textsOne, textsTwo):
    n = len(keywordsOne)
    m = len(keywordsTwo)
    co_occur = pd.DataFrame(np.zeros((n,m)), index=keywordsOne, columns=keywordsTwo)
    for txt in textsOne+textsTwo:
        tokens = set(txt.lower().split())
        presentOne = [kw for kw in keywordsOne if kw in tokens]
        presentTwo = [kw for kw in keywordsTwo if kw in tokens]
        for i in range(len(presentOne)):
            for j in range(len(presentTwo)):
                co_occur.loc[presentOne[i], presentTwo[j]] += 1
                # co_occur.loc[presentTwo[j], presentOne[i]] += 1
    pairs = []
    alpha = 1.0
    for i in range(n):
        for j in range(m):
            A, B = keywordsOne[i], keywordsTwo[j]
            score = freqsOne[A]*freqsTwo[B] - alpha * co_occur.loc[A,B]
            pairs.append({"A": A, "B": B, "gapScore": float(score), "coOccur": int(co_occur.loc[A,B])})
    pairs_sorted = sorted(pairs, key=lambda x: x["gapScore"], reverse=True)
    return pairs_sorted[:10]  # top 10 gaps



def agent(topic1, topic2, timeFrame):
    #Timeframe is months, so we need to calculate the date range
    n_months_ago_str = timeFrame.replace("-", "")+"0000"#We use military time here, hence why we need 0000
    today_str = str(datetime.now().year) + str(datetime.now().month).zfill(2) + str(datetime.now().day).zfill(
        2) + "0000"

    responseOne = requests.get(f"http://export.arxiv.org/api/query?search_query=all:{topic1}+AND+submittedDate:[{n_months_ago_str}+TO+{today_str}]&start=0&max_results=700")
    responseTwo = requests.get(f"http://export.arxiv.org/api/query?search_query=all:{topic2}+AND+submittedDate:[{n_months_ago_str}+TO+{today_str}]&start=0&max_results=700")
    # print("fetched abstracts")
    dataOne = xmltodict.parse(responseOne.text)
    dataTwo = xmltodict.parse(responseTwo.text)
    # print("parsed abstracts")
    corpusOne = [i["summary"] for i in dataOne['feed']['entry']]
    corpusTwo = [i["summary"] for i in dataTwo['feed']['entry']]
    # print("extracted summaries")
    combined_corpus = corpusOne + corpusTwo
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(combined_corpus)
    feature_names = np.array(vectorizer.get_feature_names_out())
    # print("vectorized corpus")
    # For corpusOne
    tfidf_corpusOne = tfidf_matrix[:len(corpusOne)]
    scores_one = np.asarray(tfidf_corpusOne.sum(axis=0)).flatten()
    top100_idx_one = scores_one.argsort()[::-1][:100]
    top100_terms_one = feature_names[top100_idx_one]
    corpusOneFreqs = {}
    for term in top100_terms_one:
        corpusOneFreqs[term] = sum(1 for txt in corpusOne if term in txt.lower().split())
    # print("extracted top terms for corpusOne")
    # For corpusTwo
    tfidf_corpusTwo = tfidf_matrix[len(corpusOne):]
    scores_two = np.asarray(tfidf_corpusTwo.sum(axis=0)).flatten()
    top100_idx_two = scores_two.argsort()[::-1][:100]
    top100_terms_two = feature_names[top100_idx_two]
    corpusTwoFreqs = {}
    for term in top100_terms_two:
        corpusTwoFreqs[term] = sum(1 for txt in corpusTwo if term in txt.lower().split())
    topPairs = compute_gap_scores(top100_terms_one, top100_terms_two, corpusOneFreqs, corpusTwoFreqs, corpusOne, corpusTwo)
    pairsList = "\n".join([f"{pair['A']} - {pair['B']}" for pair in topPairs])
    # print(pairsList)
    response = client.responses.create(
        model="o4-mini",
        reasoning={
            "effort": "high"
        },
        input=[
            {
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Generate multiple novel and well-developed scientific research ideas by incorporating two given keywords and the topics the user entered. Each idea must use its specific pair of keywords without mixing with keywords from other pairs and pertain to BOTH the listed topics. The goal is to outline a unique research concept that creatively combines the concepts represented by the keywords.\n\n# Steps\n\n1. **Understand the Keywords**: Analyze each set of paired keywords separately to understand their individual significance and relevant scientific contexts.\n2. **Identify Connections**: Explore potential connections or overlaps within each pair of keywords. Consider how these connections might contribute to new scientific inquiries.\n3. **Research Exploration**: Propose a research question or hypothesis that integrates both keywords from each pair in a meaningful way.\n4. **Methodological Approach**: Suggest possible methodologies, experiments, or analyses that could be undertaken to explore the research question for each keyword pair.\n5. **Potential Impact**: Discuss the potential significance or benefits of the research for each pair, such as advancements in the field, practical applications, or implications for future study.\n\n# Output Format\n\nThe output should consist of structured paragraphs for each keyword pair that includes:\n- A brief introduction of the research idea.\n- Explanation of how both keywords within the pair are integrated into the research question.\n- Description of potential methodologies or experimental approaches.\n- Discussion of the expected impact or importance of the research.\n\n# Examples\n\n**Keywords**: [nanotechnology, climate change]\n\n**Research Idea**: \nThe research idea focuses on developing advanced nanomaterials to mitigate climate change by enhancing carbon capture efficiency. By integrating the concepts of nanotechnology and environmental science, the study proposes creating porous nanomaterials that exhibit increased surface area and reactivity for capturing atmospheric CO2. Experimental approaches would involve synthesizing different nanomaterial compositions and testing their performance in controlled environment chambers. The potential impact of this research could lead to significant breakthroughs in carbon sequestration technologies, thus contributing to global efforts in reducing greenhouse gas levels.\n\n# Notes\n\n- Ensure that each research idea is novel and not a replication of existing studies.\n- Avoid overly broad or vague research concepts; focus on specificity and feasibility.\n- Consider interdisciplinary approaches to strengthen the research proposal, but maintain the integrity of keyword pairs."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Topic 1: {topic1}\nTopic 2: {topic2}\nKeyword pairs:\n{pairsList}"
                    }
                ]
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "scientific_experiment_ideas",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "ideas": {
                            "type": "array",
                            "description": "A list of ideas for scientific experiments.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {
                                        "type": "string",
                                        "description": "The title of the scientific experiment idea."
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "A basic description of what is being explored."
                                    },
                                    "keyword1": {
                                        "type": "string",
                                        "description": "The first keyword given as inspiration."
                                    },
                                    "keyword2": {
                                        "type": "string",
                                        "description": "The second keyword given as inspiration."
                                    },
                                    "hypothesis": {
                                        "type": "string",
                                        "description": "A fully fleshed-out hypothesis related to the experiment."
                                    },
                                    "procedure": {
                                        "type": "string",
                                        "description": "The experimental procedure to be followed. Remember to include newlines between steps."
                                    }
                                },
                                "required": [
                                    "title",
                                    "description",
                                    "keyword1",
                                    "keyword2",
                                    "hypothesis",
                                    "procedure"
                                ],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": [
                        "ideas"
                    ],
                    "additionalProperties": False
                }
            }
        },
        tools=[
            {
                "type": "function",
                "name": "search_arxiv_abstracts",
                "description": "Search Arxiv abstracts based on a query parameter",
                "parameters": {
                    "type": "object",
                    "required": [
                        "query",
                        "max_results",
                        "sort_by"
                    ],
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search term or phrase to query the Arxiv abstracts"
                        },
                        "max_results": {
                            "type": "number",
                            "description": "Maximum number of results to return"
                        },
                        "sort_by": {
                            "type": "string",
                            "description": "Field to sort results by, e.g., date, relevance",
                            "enum": [
                                "date",
                                "relevance"
                            ]
                        }
                    },
                    "additionalProperties": False
                },
                "strict": True
            }
        ],
        store=True
    )
    # print(response)
    resText = response.output[1].content[0].text
    return resText

# if __name__ == "__main__":
#     # Example usage
#     topic1 = "Artificial Intelligence"
#     topic2 = "Sustainability"
#     timeFrame = 12
#
#     result = agent(topic1, topic2, timeFrame)
#     print(result)  # This will print the result of the agent function when implemented.