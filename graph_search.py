from typing import List, Tuple, Dict, Optional, Any
from collections import deque
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from graph_storage import GraphStorage
from graph_entity import GraphEntity
from embedding_model import EmbeddingModel


class GraphSearch:
    """Search manager, handles all operations related to search"""

    def __init__(self, storage: GraphStorage, entity_manager: GraphEntity):
        """
        Initialize search manager

        Args:
            storage: Store a GraphSearch instance
            entity_manager: An instance of GraphEntity
        """
        self.storage = storage
        self.entity_manager = entity_manager

    def search_vector_store(
        self, query: str, entity_id: Optional[str] = None, k: int = 3
    ) -> List[Tuple[Any, float]]:
        """
        Search in vector store

        Args:
            query: Search query
            entity_id: Optional entity ID to restrict search
            k: Number of results to return

        Returns:
            List[Tuple[Any, float]]: Search results and similarity score
        """
        try:
            # If an entity ID is specified, search in the entity's vector store
            if entity_id:
                main_id = self.entity_manager._get_main_id(entity_id)
                if not main_id or main_id not in self.storage.vector_stores:
                    return []
                vector_store = self.storage.vector_stores[main_id]
            # Otherwise, search in the global vector store
            else:
                if not self.storage.global_vector_store:
                    return []
                vector_store = self.storage.global_vector_store

            # Generate query vector
            if not isinstance(query, str):
                raise ValueError("Query must be a string")

            # Excecute similarity search
            results = vector_store.similarity_search_with_score(query, k=k)

            # Process and filter results
            valid_results = []
            for doc, score in results:
                if hasattr(doc, "page_content"):
                    valid_results.append((doc.page_content, float(score)))

            return valid_results

        except Exception as e:
            print(f"[ERROR] An error occurs while searching a vector store: {str(e)}")
            return []

    def search_similar_entities(
        self, query_entity: str, top_n: int = 5, threshold: float = 0.8
    ) -> List[Tuple[str, float]]:
        """
        Search for entities similar to the given entity

        Args:
            query_entity: Entity query
            top_n: Number of results to return
            threshold: Minimum similarity score to return a result

        Returns:
            List[Tuple[str, float]]: A list in the format of (Entity ID, similarity score)
        """
        try:
            # Generate query vector
            query_embedding = EmbeddingModel.get_instance().embed_query(query_entity)
            if not isinstance(query_embedding, np.ndarray):
                query_embedding = np.array(query_embedding)

            # Compute similarity
            similarities = []
            for entity_id, entity_embedding in self.storage.entity_embeddings.items():
                if not isinstance(entity_embedding, np.ndarray):
                    entity_embedding = np.array(entity_embedding)
                similarity = cosine_similarity([query_embedding], [entity_embedding])[
                    0
                ][0]
                if similarity >= threshold:
                    similarities.append((entity_id, similarity))

            # Sort by similarity
            return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

        except Exception as e:
            print(f"[ERROR] An error occurs while searching for similar entities: {str(e)}")
            return []

    def search_similar_relationships(
        self, query: str, entity_id: str, k: int = 3
    ) -> List[Tuple[str, str, str, float]]:
        """
        Search for relationships similar to the query

        Args:
            query: Search query
            entity_id: ID of an entity
            k: the number of results to return

        Returns:
            List[Tuple[str, str, str, float]]: (实体1, 关系, 实体2, 分数)列表
        """
        main_id = self.entity_manager._get_main_id(entity_id)
        if not main_id:
            return []

        try:
            query_embedding = EmbeddingModel.get_instance().embed_query(query)
            results = []
            processed_entities = set()

            def process_entity_relationships(
                entity: str,
            ) -> List[Tuple[str, str, str, float]]:
                """Process relationships of an entity"""
                relations = []

                # Process out edges
                for successor in self.storage.graph.successors(entity):
                    edges_data = self.storage.graph.get_edge_data(entity, successor)
                    if edges_data:
                        for edge_data in edges_data.values():
                            relation_text = (
                                f"{entity} and {successor}'s relationship is {edge_data['type']}"
                            )
                            relation_embedding = (
                                EmbeddingModel.get_instance().embed_query(relation_text)
                            )
                            similarity = cosine_similarity(
                                [query_embedding], [relation_embedding]
                            )[0][0]
                            if similarity >= 0.5:
                                relations.append(
                                    (entity, edge_data["type"], successor, similarity)
                                )

                # Process in edges
                for predecessor in self.storage.graph.predecessors(entity):
                    edges_data = self.storage.graph.get_edge_data(predecessor, entity)
                    if edges_data:
                        for edge_data in edges_data.values():
                            relation_text = (
                                f"{predecessor} and {entity}'s relationship is '{edge_data['type']}"
                            )
                            relation_embedding = (
                                EmbeddingModel.get_instance().embed_query(relation_text)
                            )
                            similarity = cosine_similarity(
                                [query_embedding], [relation_embedding]
                            )[0][0]
                            if similarity >= 0.5:
                                relations.append(
                                    (predecessor, edge_data["type"], entity, similarity)
                                )

                return relations

            # First process relationships of the main entity
            results.extend(process_entity_relationships(main_id))
            processed_entities.add(main_id)

            # If no more than k results returned, search for relationships of similar entities
            while len(results) < k:
                similar_entities = []
                main_embedding = self.storage.entity_embeddings[main_id]

                # Search for similar entities
                for entity, embedding in self.storage.entity_embeddings.items():
                    if entity not in processed_entities:
                        similarity = cosine_similarity([main_embedding], [embedding])[
                            0
                        ][0]
                        if similarity >= 0.8:
                            similar_entities.append((entity, similarity))

                if not similar_entities:
                    break

                # Process similar entities
                similar_entities.sort(key=lambda x: x[1], reverse=True)
                found_new_relations = False

                for similar_entity, _ in similar_entities:
                    new_relations = process_entity_relationships(similar_entity)
                    if new_relations:
                        results.extend(new_relations)
                        found_new_relations = True
                    processed_entities.add(similar_entity)

                    if len(results) >= k:
                        break

                if not found_new_relations:
                    break

            # Return the top k results sorted by similarity
            return sorted(results, key=lambda x: x[3], reverse=True)[:k]

        except Exception as e:
            print(f"[ERROR] Error while searching for similar relationship: {str(e)}")
            return []

    def search_all_paths(
        self,
        start_entity: str,
        end_entity: str,
        max_depth: int = 5,
        max_results: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Search for the paths between two entities, prioritizing shorter paths

        Args:
            start_entity: starting entity
            end_entity: target entity
            max_depth: maximum depth to search, default is 5
            max_results: maximum number of results to return, default is 3

        Returns:
            List[Dict[str, Any]]: List of paths information sorted by path length, each dictionary contains:
                - path: list of entity IDs on the path
                - relationships: relationship descriptions on the path
                - length: length of the path
        """
        start_main_id = self.entity_manager._get_main_id(start_entity)
        end_main_id = self.entity_manager._get_main_id(end_entity)

        if not start_main_id or not end_main_id or start_main_id == end_main_id:
            return []

        # list of results
        all_paths = []

        # For BFS queue: (current node, current path, relationship list, current depth)
        queue = deque([(start_main_id, [start_main_id], [], 0)])

        # Visiting history: (node, depth) -> whether it is visited
        # The same node can be visited at different depths, but shorter paths should be visited first
        visited = set()

        while queue and len(all_paths) < max_results:
            current, current_path, relations, depth = queue.popleft()

            # Skip if exceeds max depth
            if depth > max_depth:
                continue

            # Marker for current state
            state = (current, depth)
            if state in visited:
                continue
            visited.add(state)

            # If the target node is found
            if current == end_main_id:
                path_info = {
                    "path": current_path,
                    "relationships": relations,
                    "length": len(current_path) - 1,
                }
                all_paths.append(path_info)
                continue

            # Get all neighboring nodes
            neighbors = []

            # Process out edges
            for successor in self.storage.graph.successors(current):
                edges_data = self.storage.graph.get_edge_data(current, successor)
                for edge_data in edges_data.values():
                    neighbors.append(("out", successor, edge_data["type"]))

            # Process in edges
            for predecessor in self.storage.graph.predecessors(current):
                edges_data = self.storage.graph.get_edge_data(predecessor, current)
                for edge_data in edges_data.values():
                    neighbors.append(("in", predecessor, edge_data["type"]))

            # Traverse all neighboring nodes
            for direction, next_node, relation_type in neighbors:
                # Check if the next state has been visited
                next_state = (next_node, depth + 1)
                if next_state in visited:
                    continue

                # Construct relationship description
                if direction == "out":
                    relation_desc = f"{current} -{relation_type}-> {next_node}"
                else:
                    relation_desc = f"{next_node} -{relation_type}-> {current}"

                # Construct new path and relationship list
                new_path = current_path + [next_node]
                new_relations = relations + [relation_desc]

                # Add new state to the queue
                queue.append((next_node, new_path, new_relations, depth + 1))

        return all_paths  # Since we are using BFS, it's already ranked by length

    def tree_search(self, start_entity: str, max_depth: int = 3) -> nx.DiGraph:
        """
        Start tree search from the starting entity

        Args:
            start_entity: starting entity ID
            max_depth: maximum depth to search

        Returns:
            nx.DiGraph: a directed graph representing the search tree
        """
        start_main_id = self.entity_manager._get_main_id(start_entity)
        if start_main_id:
            return nx.bfs_tree(self.storage.graph, start_main_id, depth_limit=max_depth)
        return nx.DiGraph()

    def search_communities(
        self, query: str, top_n: int = 1, threshold: float = 0.5
    ) -> List[Tuple[List[str], str]]:
        """
        Based on the query, search for related communities and return the list of entities in the community and the community summary

        Args:
            query: user query string
            top_n: maximum number of communities to return
            threshold: similarity threshold, only return results if the score is higher than this value

        Returns:
            List[Tuple[List[str], str]]: Each tuple contains (list of community entities, community summary).
            Return an empty list when all results have similarity scores lower than the threshold.
        """
        if not self.storage.community_vector_store:
            return []

        try:
            # Perform similarity search
            results = self.storage.community_vector_store.similarity_search_with_score(
                query, k=top_n
            )
            communities_data = []

            for doc, score in results:
                # Process the result only if the similarity score is higher than the threshold
                if score > threshold:
                    # Extract community ID from metadata
                    community_id = doc.metadata["Community"].split("_")[
                        1
                    ]  # Extract '25' from 'Community_25'

                    # Extract relevant information from community data
                    community_data = self.storage.communities[community_id]

                    members = community_data["members"]
                    summary = community_data["summary"]
                    communities_data.append((members, summary))

            return communities_data

        except Exception as e:
            print(f"[ERROR] Error while searching communities: {str(e)}")
            return []
