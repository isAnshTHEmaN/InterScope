import json

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

def invoke_functions_from_response(response):
    """Extract all function calls from the response, look up the corresponding tool function(s) and execute them.
    (This would be a good place to handle asynchroneous tool calls, or ones that take a while to execute.)
    This returns a list of messages to be added to the conversation history.
    """
    intermediate_messages = []
    for response_item in response.output:
        if response_item.type == 'function_call':
            try:
                arguments = json.loads(response_item.arguments)
                print(f"Invoking tool: {response_item.name}({arguments})")
                response = requests.get(f"http://export.arxiv.org/api/query?search_query=all:{arguments['query']}&start=0&max_results={arguments['max_results']}")
                tool_output = response.text
            except Exception as e:
                msg = f"Error executing function call: {response_item.name}: {e}"
                tool_output = msg
                print(msg)
            intermediate_messages.append({
                "type": "function_call_output",
                "call_id": response_item.call_id,
                "output": tool_output
            })
        elif response_item.type == 'reasoning':
            print(f'Reasoning step: {response_item.summary}')
    return intermediate_messages

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
    tools = [{
        "type": "function",
        "name": "search_arxiv_abstracts",
        "description": "Search Arxiv abstracts based on a query parameter",
        "parameters": {
            "type": "object",
            "required": [
                "query",
                "max_results",
            ],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search term or phrase to query the Arxiv abstracts"
                },
                "max_results": {
                    "type": "number",
                    "description": "Maximum number of results to return"
                }
            },
            "additionalProperties": False
        },
        "strict": True
    }]
    outputSchema = {
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
    messages = [
        {
            "role": "developer",
            "content": [
                {
                    "type": "input_text",
                    "text": "Generate multiple novel and well-developed scientific research ideas by incorporating two given keywords and the topics the user entered. Each idea must use its specific pair of keywords without mixing with keywords from other pairs and pertain to BOTH the listed topics. The goal is to outline a unique research concept that creatively combines the concepts represented by the keywords.\n\n# Steps\n\n1. **Understand the Keywords**: Analyze each set of paired keywords separately to understand their individual significance and relevant scientific contexts.\n2. **Identify Connections**: Explore potential connections or overlaps within each pair of keywords. Consider how these connections might contribute to new scientific inquiries.\n3. **Research Exploration**: Propose a research question or hypothesis that integrates both keywords from each pair in a meaningful way.\n4. **Methodological Approach**: Suggest possible methodologies, experiments, or analyses that could be undertaken to explore the research question for each keyword pair.\n5. **Potential Impact**: Discuss the potential significance or benefits of the research for each pair, such as advancements in the field, practical applications, or implications for future study.\n\n# Output Format\n\nThe output should consist of structured paragraphs for each keyword pair that includes:\n- A brief introduction of the research idea.\n- Explanation of how both keywords within the pair are integrated into the research question.\n- Description of potential methodologies or experimental approaches.\n- Discussion of the expected impact or importance of the research.\n\n# Examples\n\n**Keywords**: [nanotechnology, climate change]\n\n**Research Idea**: \nThe research idea focuses on developing advanced nanomaterials to mitigate climate change by enhancing carbon capture efficiency. By integrating the concepts of nanotechnology and environmental science, the study proposes creating porous nanomaterials that exhibit increased surface area and reactivity for capturing atmospheric CO2. Experimental approaches would involve synthesizing different nanomaterial compositions and testing their performance in controlled environment chambers. The potential impact of this research could lead to significant breakthroughs in carbon sequestration technologies, thus contributing to global efforts in reducing greenhouse gas levels.\n\n# Notes\n\n- Ensure that each research idea is novel and not a replication of existing studies.\n- Avoid overly broad or vague research concepts; focus on specificity and feasibility.\n- Consider interdisciplinary approaches to strengthen the research proposal, but maintain the integrity of keyword pairs. ALWAYS use arxiv search to help you brainstorm and gague the novelty of your ideas."
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
    ]
    response = client.responses.create(
        model="o4-mini",
        reasoning={
            "effort": "high"
        },
        input=messages,
        text={
            "format": outputSchema
        },
        tools=tools,
        store=True
    )
    while True:
        function_responses = invoke_functions_from_response(response)
        if len(function_responses) == 0:  # We're done reasoning
            print(response.output_text)
            break
        else:
            print("More reasoning required, continuing...")
            response = client.responses.create(
                model="o4-mini",
                reasoning={
                    "effort": "high"
                },
                input=messages,
                text={
                    "format": outputSchema
                },
                tools=tools,
                store=True
            )
    resText = response.output[1].content[0].text
    return resText

if __name__ == "__main__":
    # Example usage
    topic1 = "Artificial Intelligence"
    topic2 = "Sustainability"
    timeFrame = "2024-01-01"  # YYYY-MM-DD format

    result = agent(topic1, topic2, timeFrame)
    print(result)  # This will print the result of the agent function when implemented.
