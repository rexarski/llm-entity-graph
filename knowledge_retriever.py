from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import json
import re
from knowledgeGraph import KnowledgeGraph


class RetrievalMode(Enum):
    FAST = "1"  # Fast retrieval: find the most relevant content directly
    ASSOCIATE = "2"  # Associate retrieval: expand search based on initial retrieval results
    RELATION = "3"  # Relation retrieval: focus on the relationship network between entities
    COMMUNITY = "4"  # Community retrieval: find related discussions in the community


class KnowledgeRetriever:
    def __init__(self, knowledge_base_path: str = "./knowledge_base"):
        """Initialize the knowledge retrieval service"""
        self.retrieval_cache: Dict[str, int] = {}  # Retrieval result cache
        self.initial_cache_rounds = 5  # Initial rounds for caching

        try:
            print(f"\n[Info] Loading knowledge graph...")
            self.kg = KnowledgeGraph(knowledge_base_path)
            print(f"[Info] Knowledge graph loaded！")
        except Exception as e:
            print(f"[ERROR] Knowledge graph loading error: {str(e)}")
            self.kg = None

    def _get_cached_results(self, results: List[Tuple[Any, float]]) -> List[str]:
        """Get cached retrieval results
        Return the first content that is not in the cache, or the first content if all are cached
        """
        if not results:
            return []

        # Process the first result
        first_content = results[0][0]
        content = (
            first_content
            if isinstance(first_content, str)
            else first_content.page_content
        )

        # Return the first content if it is not in the cache
        if content not in self.retrieval_cache:
            self.retrieval_cache[content] = self.initial_cache_rounds
            return [content]

        # If the first content is in the cache, increase the count
        self.retrieval_cache[content] += 1

        # Iterate through the remaining results to find uncached content
        for doc, _ in results[1:]:
            next_content = doc if isinstance(doc, str) else doc.page_content
            if next_content not in self.retrieval_cache:
                # Found uncached content, add to cache and return
                self.retrieval_cache[next_content] = self.initial_cache_rounds
                return [next_content]
            else:
                # Cached content, increase count
                self.retrieval_cache[next_content] += 1

        # If all content is cached, return the first content
        return [content]

    def _cleanup_cache(self):
        """Clean up cache"""
        expired_entries = [
            content for content, rounds in self.retrieval_cache.items() if rounds <= 0
        ]

        for content in expired_entries:
            del self.retrieval_cache[content]

    def update_cache_counts(self):
        """Update cache counts"""
        for content in list(self.retrieval_cache.keys()):
            self.retrieval_cache[content] -= 1
        self._cleanup_cache()

    def parse_query_response(self, response: str) -> Dict:
        """Parse the query response from LLM"""
        default_response = {"query": "", "entities": [], "reply": ""}

        try:
            json_match = re.search(r"```json\s*([\s\S]*?)\s*```", response)
            json_str = json_match.group(1) if json_match else response
            query_info = json.loads(json_str)

            if all(key in query_info for key in ["query", "entities"]):
                return query_info
            else:
                print("[Info] Missing required fields in query response")
                return default_response

        except Exception as e:
            print(f"[Info] Parsing error: {str(e)}")
            return default_response

    def fast_retrieval(self, query: str) -> Optional[str]:
        """Fast retrieval: search directly in the vector store"""
        try:
            results = self.kg.search_vector_store(query, k=5)
            filtered_results = [(doc, score) for doc, score in results if score >= 0.55]

            if filtered_results:
                selected_contents = self._get_cached_results(filtered_results)
                return "\n\n".join(selected_contents)

        except Exception as e:
            print(f"[ERROR] Fast retrieval error: {str(e)}")
        return None

    def associate_retrieval(self, query: str, entities: List[str]) -> Optional[str]:
        """
        Associated retrieval: expand search based on entities and relationships network
        - If results are found in the specified entity vector store, continue with associated search
        - If no results are found in the specified entity vector store, search in the global vector store
        """
        try:
            retrieval_results = []
            retrieved_contents = set()  # Keep track of retrieved content to avoid duplication
            found_in_entity_store = False  # Mark whether results are found in the entity vector store

            for entity in entities:
                # Search for similar entities
                similar_entities = self.kg.search_similar_entities(
                    entity, top_n=1, threshold=0.8
                )
                if not similar_entities:
                    continue

                main_entity = similar_entities[0][0]

                # Search in the main entity's vector store
                entity_results = self.kg.search_vector_store(
                    query, entity_id=main_entity, k=5
                )
                filtered_results = [
                    (doc, score) for doc, score in entity_results if score >= 0.5
                ]

                if filtered_results:
                    found_in_entity_store = True
                    selected_contents = self._get_cached_results(filtered_results)
                    for content in selected_contents:
                        # Check if the content already exists
                        if content not in retrieved_contents:
                            retrieved_contents.add(content)
                            retrieval_results.append(
                                f"[{main_entity}]-related content：\n{content}"
                            )

            # If no results are found in the entity vector store, search in the global vector store and return
            if not found_in_entity_store:
                global_results = self.kg.search_vector_store(query, k=5)
                filtered_global_results = [
                    (doc, score) for doc, score in global_results if score >= 0.5
                ]

                if filtered_global_results:
                    selected_contents = self._get_cached_results(
                        filtered_global_results
                    )
                    for content in selected_contents:
                        if content not in retrieved_contents:
                            retrieved_contents.add(content)
                            retrieval_results.append(f"[Global search]Related content：\n{content}")

                return "\n\n".join(retrieval_results) if retrieval_results else None

            # If results are found in the entity vector store, continue with associated entity search
            for entity in entities:
                similar_entities = self.kg.search_similar_entities(
                    entity, top_n=1, threshold=0.85
                )
                if not similar_entities:
                    continue

                main_entity = similar_entities[0][0]

                # Get similar relationships
                relationships = self.kg.search_similar_relationships(
                    query, main_entity, k=3
                )
                if not relationships:
                    continue

                # Search in related entities
                related_entities = set()  # Keep track of related entities
                relations_added = set()  # Keep track of added relationship descriptions
                entity_relations = {}  # Keep track of relationship descriptions for entities

                for source, relation, target, score in relationships:
                    # Construct a complete relationship query statement
                    relation_query = f"The relation between {source} and {target} is: {relation}"

                    # Keep track of relationship information
                    if relation_query not in relations_added:
                        relations_added.add(relation_query)
                        retrieval_results.append(f"[Related Relationship]：\n- {relation_query}")

                    # Create a set of entities in the relationship (excluding the main entity) and record the corresponding relationship description
                    if source != main_entity:
                        related_entities.add(source)
                        entity_relations[source] = relation_query
                    if target != main_entity:
                        related_entities.add(target)
                        entity_relations[target] = relation_query

                # Search in related entities using relationship query statements
                for related_entity in related_entities:
                    relation_query = entity_relations[related_entity]
                    results = self.kg.search_vector_store(
                        query=relation_query, entity_id=related_entity, k=5
                    )
                    filtered_results = [
                        (doc, score) for doc, score in results if score >= 0.5
                    ]

                    if filtered_results:
                        selected_contents = self._get_cached_results(filtered_results)
                        for content in selected_contents:
                            # Check if the content already exists
                            content_is_unique = True
                            normalized_content = "".join(content.split())

                            for existing_content in retrieved_contents:
                                normalized_existing = "".join(existing_content.split())
                                if (
                                    normalized_content in normalized_existing
                                    or normalized_existing in normalized_content
                                ):
                                    content_is_unique = False
                                    break

                            if content_is_unique:
                                retrieved_contents.add(content)
                                retrieval_results.append(
                                    f"[Related Entity - {related_entity}]：\n{content}"
                                )

            return "\n\n".join(retrieval_results) if retrieval_results else None

        except Exception as e:
            print(f"[ERROR] Failed to conduct an associate retrieval: {str(e)}")
        return None

    def relation_retrieval(self, entities: List[str]) -> Optional[str]:
        """
        Scour the entity list for relationships between each pair of entities, and perform vector retrieval on the first path
        """
        try:
            if len(entities) < 2:
                return None

            # 1. Entity matching
            main_entities = []
            matched_indices = []  # Keep track of the original entity index that was successfully matched

            for i, entity in enumerate(entities):
                similar_entities = self.kg.search_similar_entities(
                    entity, top_n=1, threshold=0.85
                )
                if similar_entities:
                    main_entities.append(similar_entities[0][0])
                    matched_indices.append(i)

            if len(main_entities) < 2:
                return None

            result_parts = []
            seen_paths = set()  # For path deduplication
            seen_contents = set()  # For content deduplication

            # 2. Search for paths between matched entities
            for i in range(len(main_entities)):
                for j in range(i + 1, len(main_entities)):
                    # Use matched entities and corresponding original entity indices
                    entity1 = main_entities[i]
                    entity2 = main_entities[j]
                    original_entity1 = entities[matched_indices[i]]
                    original_entity2 = entities[matched_indices[j]]

                    # Prevent duplicate paths
                    path_key = f"{min(entity1, entity2)}-{max(entity1, entity2)}"
                    if path_key in seen_paths:
                        continue
                    seen_paths.add(path_key)

                    # Search all paths between two entities
                    paths = self.kg.search_all_paths(entity1, entity2, max_depth=5)

                    if paths:
                        result_parts.append(
                            f"\n{original_entity1} - {original_entity2}'s relationship:"
                        )

                        # Display all paths
                        for path_idx, path_info in enumerate(paths, 1):
                            result_parts.append(f"Path {path_idx}:")
                            result_parts.append(
                                f"Entity path: {' -> '.join(path_info['path'])}"
                            )
                            result_parts.append("Chain of relationships:")
                            result_parts.extend(
                                f"  {rel}" for rel in path_info["relationships"]
                            )

                            # Only perform vector search on the first path
                            if path_idx == 1:
                                for relationship in path_info["relationships"]:
                                    try:
                                        start_end = relationship.split("->")
                                        if len(start_end) == 2:
                                            start_part = start_end[0].strip()
                                            end_entity = start_end[1].strip()

                                            start_relation = start_part.split("-", 1)
                                            if len(start_relation) == 2:
                                                start_entity = start_relation[0].strip()
                                                relation = start_relation[1].strip()

                                                relation_query = f"The relationship between {start_entity} and {end_entity} is: {relation}"

                                                entity_results = (
                                                    self.kg.search_vector_store(
                                                        query=relation_query,
                                                        entity_id=start_entity,
                                                        k=3,
                                                    )
                                                )

                                                filtered_results = [
                                                    (doc, score)
                                                    for doc, score in entity_results
                                                    if score >= 0.5
                                                ]
                                                if filtered_results:
                                                    selected_contents = (
                                                        self._get_cached_results(
                                                            filtered_results
                                                        )
                                                    )
                                                    for content in selected_contents:
                                                        normalized_content = "".join(
                                                            content.split()
                                                        )
                                                        if (
                                                            normalized_content
                                                            not in seen_contents
                                                        ):
                                                            seen_contents.add(
                                                                normalized_content
                                                            )
                                                            result_parts.append(
                                                                f"[{start_entity}->{end_entity}] related content：\n{content}\n\n"
                                                            )

                                    except Exception as e:
                                        print(
                                            f"[ERROR] Failed to process the relationship '{relationship}': {str(e)}"
                                        )
                                        continue

                        result_parts.append("-" * 50)  # Add a divider

            return "\n".join(result_parts) if result_parts else None

        except Exception as e:
            print(f"[ERROR] Failed at relationship retrieval: {str(e)}")
        return None

    def community_retrieval(self, query: str) -> Optional[str]:
        """
        Community retrieval: find related community information and relevant content in global documents

        Args:
            query: user's query string

        Returns:
            Optional[str]: Returns the retrieval result, including community information and related content. Returns None if no relevant content is found
        """
        try:
            result_parts = []

            # 1. Community retrieval
            community_results = self.kg.search_communities(query, top_n=1)
            if community_results:
                members, summary = community_results[0]
                result_parts.append("【请参考社区观点】")
                result_parts.append("相关社区成员:")
                result_parts.append(f"- {', '.join(members)}")
                result_parts.append("\n社区简介:")
                result_parts.append(summary)

            # 2. Global document retrieval
            doc_results = self.kg.search_vector_store(query, k=5)
            filtered_results = [
                (doc, score) for doc, score in doc_results if score >= 0.5
            ]

            if filtered_results:
                selected_contents = self._get_cached_results(filtered_results)
                if selected_contents:
                    if result_parts:  # If there are community results before, add a separator
                        result_parts.append("\n")
                    result_parts.append(
                        "[The following information is for reference only. Please use the community opinions above in your response]"
                    )
                    result_parts.extend(selected_contents)

            # Return if any retrieval results are found
            if result_parts:
                return "\n".join(result_parts)

        except Exception as e:
            print(f"[ERROR] Failed to conduct the community retrieval: {str(e)}")
        return None

    def retrieve(
        self, mode: RetrievalMode, query: str, entities: List[str]
    ) -> Optional[str]:
        """Unified retrieval interface"""
        if not self.kg:
            return None

        self.update_cache_counts()

        try:
            if mode == RetrievalMode.FAST:
                return self.fast_retrieval(query)

            elif mode == RetrievalMode.ASSOCIATE:
                return self.associate_retrieval(query, entities)

            elif mode == RetrievalMode.RELATION:
                if len(entities) >= 2:
                    return self.relation_retrieval(entities)

            elif mode == RetrievalMode.COMMUNITY:
                return self.community_retrieval(query)

        except Exception as e:
            print(f"[ERROR] Failed to retrieve: {str(e)}")

        return None
