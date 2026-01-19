VERIFY_PROMPT = """
### Task
- Based on the provided graph triplets, verify whether the fact in the following sentence is true or false. Only ground on the knowledge in the given triplets.
- The provided triplets are all the relevant information from the knowledge graph, if the fact is provided as negation and all the triplets does not contain the fact, the fact is true.
- You do NOT have to answer if the sentence is a question only verify the fact in that question. For example, you only have to verify 'Daniel Martínez (politician) a leader of Montevideo' in the sentence 'When was Daniel Martínez (politician) a leader of Montevideo?'
- Note that the "~" symbol indicates the reverse relation. For example, "A ~loves B" means "B loves A" or "A ~south of B" means "B is north of A".

### Respond in the following JSON format without additional information:
{
    "rationale": "a short rationale for the decision",
    "verdict": "true/false as a json value"
}

### Triplets:
{{triplets}}

### Sentence: {{claim}}
""".strip()

VERIFY_PROMPT = """
### Task:
Verify whether the fact stated in the given sentence is true or false based solely on the provided graph triplets. Use only the information in the triplets for verification.

- The triplets provided represent all relevant knowledge from the knowledge graph.
- If the fact is a negation and the triplets do not include the fact, consider the fact as true.
- Ignore questions and verify only the factual assertion within them. For example, in the question 'When was Daniel Martínez (politician) a leader of Montevideo?', focus on verifying the assertion 'Daniel Martínez (politician) a leader of Montevideo'.
- Interpret the "~" symbol in triplets as indicating a reverse relationship. For example:
  - "A ~loves B" means "B loves A".
  - "A ~south of B" means "B is north of A".

### Response Format:
Provide your response in the following JSON format without any additional explanations:

{
    "rationale": "A concise explanation for your decision",
    "verdict": "true/false as the JSON value"
}

### Triplets:
{{triplets}}

### Sentence:
{{claim}}
""".strip()
