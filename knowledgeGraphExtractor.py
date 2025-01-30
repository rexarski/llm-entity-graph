import os
import sys
import json
import re
# from openai import OpenAI
from ollama import Client
from pydantic import BaseModel
from tqdm import tqdm
from knowledgeGraph import KnowledgeGraph
from config import *

# client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
client = Client(
    host='http://localhost:11434',
    # headers={'x-some-header': 'some-value'}
)

class Entities(BaseModel):
    entities: list[str]
class Relation(BaseModel):
    source: str
    target: str
    relation: str
class Relations(BaseModel):
    relations: list[Relation]

class KnowledgeGraphExtractor:
    def __init__(self):
        # Initialize the KnowledgeGraph class
        self.kg = KnowledgeGraph("./knowledge_base")

        # Read the prompt templates
        self.entity_prompt = self.read_prompt("prompt/entity_extraction.txt")
        self.relation_prompt = self.read_prompt("prompt/relationship_extraction.txt")

        # Load the progress file
        self.progress_file = "data/processed_files.txt"
        self.processed_files = self.load_progress()

    def load_progress(self):
        """List of processed files"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, "r", encoding="utf-8") as f:
                return set(line.strip() for line in f)
        return set()

    def save_progress(self, item_id):
        """Save the processed file"""
        os.makedirs(os.path.dirname(self.progress_file), exist_ok=True)
        with open(self.progress_file, "a", encoding="utf-8") as f:
            f.write(f"{item_id}\n")

    @staticmethod
    def read_prompt(file_path):
        """Read the prompt file"""
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    @staticmethod
    def read_json_file(file_path):
        """Read the JSON file"""
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def parse_ai_response(self, response):
        """Parse the AI response"""
        json_match = re.search(r"```json\s*([\s\S]*?)\s*```", response)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print(f"Failed JSON parsing: {json_str}")
            return {}

    def chat_with_LLM(self, messages, output_format):
        """Chat with the LLM model"""
        try:
            # response = client.chat.completions.create(
            #     model="moonshot-v1-32k",
            #     messages=messages,
            #     temperature=0.5,
            #     response_format={"type": "json_object"},
            #     stream=True,
            # )

            # structured output! https://ollama.com/blog/structured-outputs
            response = client.chat(
                model=LLM_MODEL_NAME,
                messages=messages,
                # temperature=0.5,
                # response_format={"type": "json_object"},
                # format="json",
                format=output_format,
                stream=True,
            )

            full_response = ""
            for chunk in response:
                # if chunk.choices[0].delta.content is not None:
                if chunk.message.content is not None:
                    # content = chunk.choices[0].delta.content
                    content = chunk.message.content
                    print(content, end="", flush=True)
                    full_response += content
            print()
            return full_response

        except Exception as e:
            print(f"Failed calling LLM: {str(e)}")
            raise

    def extract_entities(self, text):
        """Extract entities"""
        messages = [
            {"role": "system", "content": self.entity_prompt},
            {
                "role": "user",
                "content": f"Please extract core entities related to the topic from the following text：\n\n{text}",
            },
        ]
        response = self.chat_with_LLM(messages, output_format=Entities.model_json_schema())
        return self.parse_ai_response(response)

    def extract_relations(self, text, entities):
        """Extract relations"""
        entities_str = ", ".join(entities)
        messages = [
            {"role": "system", "content": self.relation_prompt},
            {
                "role": "user",
                "content": f"List of known entities：{entities_str}\n\nPlease extract the relationships between these entities from the following text:\n\n{text}",
            },
        ]

        response = self.chat_with_LLM(messages, output_format=Relations.model_json_schema())
        return self.parse_ai_response(response)

    def process_item(self, item_id, item_data):
        """Process the item data"""
        try:
            title = item_data.get("title", "")
            clusters = item_data.get("clusters", [])

            print(f"[INFO] Processing data {item_id}: {title}")
            print(f"[INFO] Data {item_id} contains {len(clusters)} text clusters")

            entity_contents = {}
            all_relations = []

            for i, cluster in enumerate(clusters):
                print(f"[INFO] Processing {i+1}/{len(clusters)} text clusters")

                comments = [
                    comment.replace("\n", " ").strip()
                    for comment in cluster.get("comments", [])
                ]
                context = f"Topic: {title}\nAll comments:\n" + "\n".join(comments)

                entities_result = self.extract_entities(context)
                entities = entities_result.get("entities", [])

                if not entities:
                    continue

                relations_result = self.extract_relations(context, entities)
                relations = relations_result.get("relations", [])

                if not relations:
                    continue

                content_unit_title = ", ".join(entities)
                content_unit = context

                for entity in entities:
                    if entity not in entity_contents:
                        entity_contents[entity] = []
                    entity_contents[entity].append((content_unit_title, content_unit))

                all_relations.extend(relations)

            if not entity_contents or not all_relations:
                print(f"[WARNING] Data {item_id} has no effective entity or relationships")
                return False

            print(
                f"[INFO] Adding {len(entity_contents)} entities and {len(all_relations)} relationships to the knowledge graph"
            )

            for entity, content_units in entity_contents.items():
                self.kg.add_entity(entity, content_units)

            for relation in all_relations:
                self.kg.add_relationship(
                    relation["source"], relation["target"], relation["relation"]
                )

            return True

        except Exception as e:
            print(f"[ERROR] Item processing error: {str(e)}")
            return False

    def process_data(self, input_file="data/results.json"):
        """Process the data"""
        try:
            data = self.read_json_file(input_file)

            unprocessed_items = [
                (item_id, data)
                for item_id, data in list(data.items())
                if item_id not in self.processed_files
            ]

            if not unprocessed_items:
                print("[INFO] No more data to process")
                return self.kg

            print(f"[INFO] Will process {len(unprocessed_items)} items")

            for item_id, item_data in tqdm(unprocessed_items):
                if self.process_item(item_id, item_data):
                    self.save_progress(item_id)
                    self.processed_files.add(item_id)
                    self.kg.save()
                    print(f"[INFO] Item {item_id} has been processed and saved")

            self.kg.merge_similar_entities()
            self.kg.remove_duplicates()
            self.kg.visualize()

            print("\n[INFO] Data processing completed")
            return self.kg

        except Exception as e:
            print(f"[ERROR] Data processing error: {str(e)}")
            raise


def main():
    try:
        print("\n[INFO] Starting data processing")
        extractor = KnowledgeGraphExtractor()
        kg = extractor.process_data()
        print("[INFO] Knowledge graph has been completed and the corresponding visualization has been saved")

    except Exception as e:
        print(f"[ERROR] Error when executing: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
