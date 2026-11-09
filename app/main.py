from app.rag_chain import chain_retrieving


"""
result = chain_retrieving.invoke({
    "lecture": "Adding a custom calculation",
    "title": "Why are we using SUM here? It's unclear to me.",
    "body": "This question refers to calculating the GM%."
})

print(result)
"""


result_streamed = chain_retrieving.stream({
    "lecture": "Adding a custom calculation",
    "title": "Why are we using SUM here? It's unclear to me.",
    "body": "This question refers to calculating the GM%."
})

for chunk in result_streamed:
    print(chunk, end="", flush=True)