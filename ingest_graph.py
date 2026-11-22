import os
import json

from dotenv import load_dotenv
from openai import OpenAI
from neo4j import GraphDatabase

# -----------------------------------------
# 1. Load environment variables
# -----------------------------------------

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# -----------------------------------------
# 2. Load chunks created in Step 2
# -----------------------------------------

CHUNKS_PATH = "vector_store/chunks.json"

with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
    chunks = json.load(file)

print(f"Loaded {len(chunks)} chunks")

# -----------------------------------------
# 3. Connect to Neo4j Aura
# -----------------------------------------

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)

driver.verify_connectivity()

print("Neo4j connection successful!")

# -----------------------------------------
# 4. Extract triples using OpenAI
# -----------------------------------------

def extract_triples(chunk_text):

    prompt = f"""
Extract factual knowledge graph triples from the text.

Return ONLY valid JSON in this format:

{{
    "triples": [
        {{
            "subject": "PostgreSQL",
            "relation": "stores",
            "object": "Users"
        }}
    ]
}}

Rules:
- Extract only facts explicitly mentioned in the text.
- Subject and object should be entities.
- Relation should be a short action or relationship.
- Do not invent information.
- If no triples are found, return an empty list.
- Avoid duplicate triples.

Text:
{chunk_text}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You extract factual knowledge graph triples."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={"type": "json_object"},
        temperature=0
    )

    result = response.choices[0].message.content

    data = json.loads(result)

    return data.get("triples", [])


# -----------------------------------------
# 5. Store triples in Neo4j
# -----------------------------------------

def store_triple(
    subject,
    relation,
    object_entity,
    source,
    page,
    chunk_id
):

    query = """
    MERGE (s:Entity {name: $subject})
    MERGE (o:Entity {name: $object})
    MERGE (s)-[r:RELATED_TO {
        relation: $relation,
        chunk_id: $chunk_id
    }]->(o)

    SET r.source = $source,
        r.page = $page

    RETURN s.name, r.relation, o.name
    """

    with driver.session(database=NEO4J_DATABASE) as session:

        session.run(
            query,
            subject=subject,
            relation=relation,
            object=object_entity,
            source=source,
            page=page,
            chunk_id=chunk_id
        )


# -----------------------------------------
# 6. Process every chunk
# -----------------------------------------

total_triples = 0

print("\nStarting knowledge graph ingestion...\n")

for chunk_id, chunk in enumerate(chunks):

    text = chunk["text"]

    source = chunk.get("source", "unknown")
    page = chunk.get("page", 0)

    print(f"Processing chunk {chunk_id + 1}/{len(chunks)}")

    try:

        triples = extract_triples(text)

        print(f"Triples extracted: {len(triples)}")

        for triple in triples:

            subject = triple.get("subject", "").strip()
            relation = triple.get("relation", "").strip()
            object_entity = triple.get("object", "").strip()

            if not subject or not relation or not object_entity:
                continue

            store_triple(
                subject=subject,
                relation=relation,
                object_entity=object_entity,
                source=source,
                page=page,
                chunk_id=chunk_id
            )

            total_triples += 1

    except Exception as e:

        print(f"Error processing chunk {chunk_id}: {e}")


# -----------------------------------------
# 7. Close connection
# -----------------------------------------

driver.close()

# -----------------------------------------
# 8. Final result
# -----------------------------------------

print("\n================================")
print("KNOWLEDGE GRAPH INGESTION COMPLETE")
print("================================")

print(f"Chunks processed: {len(chunks)}")
print(f"Triples stored: {total_triples}")

print("Knowledge graph stored in Neo4j Aura!")