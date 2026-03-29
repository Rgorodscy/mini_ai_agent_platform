import json

from app.config import groq_client


EXPAND_PROMPT = """
You are an assistant that expands search queries.
Return ONLY a JSON array with strings.
No aditional text, no markdown, no explanations.
Example: ["original query", "variation 1", "variation 2", "variation 3"]',
"""


def expand_query(query: str) -> list[str]:
    res = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": EXPAND_PROMPT,
            },
            {"role": "user", "content": query},
        ],
    )
    raw = res.choices[0].message.content.strip()
    return json.loads(raw)


if __name__ == "__main__":

    queries = expand_query("ana silva")
    print(queries)
