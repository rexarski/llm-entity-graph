import os
from typing import List, Tuple, Optional, Dict, Any
import networkx as nx
# from openai import OpenAI
from ollama import Client

from graph_storage import GraphStorage
from graph_entity import GraphEntity
from graph_search import GraphSearch
from graph_visualization import GraphVisualization
from config import API_KEY, API_BASE_URL


class KnowledgeGraph:
    """Class of the KnowledgeGraph, integrating all functional components"""

    def __init__(self, base_path: str):
        """
        Initialize the knowledge graph

        Args:
            base_path: base path of the knowledge graph data
        """
        # Initialize the LLM client
        # self.llm_client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
        self.llm_client = Client(
            host='http://localhost:11434',
        )

        # Initialize the components
        self.storage = GraphStorage(base_path)
        self.entity = GraphEntity(self.storage, self.llm_client)
        self.search = GraphSearch(self.storage, self.entity)
        self.visualization = GraphVisualization(self.storage)

        # Load existing graph
        if os.path.exists(base_path):
            self._load_existing_graph()
            print(f"Existing knowledge graph loaded: {base_path}")
        else:
            self._initialize_new_graph()
            print(f"Creaing new knowledge graph: {base_path}")

    def _load_existing_graph(self) -> None:
        """Load existing graph"""
        try:
            self.storage.load()
        except Exception as e:
            print(f"[ERROR] Failed while loading existing graph: {str(e)}")
            print("[INFO] Creating a new knowledge graph...")
            self._initialize_new_graph()

    def _initialize_new_graph(self) -> None:
        """Initialize a new graph"""
        try:
            self.storage._init_storage()
        except Exception as e:
            print(f"[ERROR] Failed to initialize a new graph: {str(e)}")

    def save(self) -> None:
        """Save the knowledge graph"""
        try:
            self.storage.save()
            print("Graph saved successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to save the graph: {str(e)}")

    # Entity management
    def add_entity(self, entity_id: str, content_units: List[Tuple[str, str]]) -> str:
        """
        Add entity to the graph

        Args:
            entity_id: Main entity ID
            content_units: A list of content units in the format of [(title, content),...]

        Returns:
            str: Main entity ID (possibly merged ID)
        """
        return self.entity.add_entity(entity_id, content_units)

    def add_relationship(
        self, entity1_id: str, entity2_id: str, relationship_type: str
    ) -> None:
        """Add relationship between entities"""
        self.entity.add_relationship(entity1_id, entity2_id, relationship_type)

    def get_entity_info(self, entity_id: str) -> Optional[Dict]:
        """Get information of the entity"""
        return self.entity.get_entity_info(entity_id)

    def get_relationships(self, entity1_id: str, entity2_id: str) -> List[str]:
        """Get all relationships between two entities"""
        return self.entity.get_relationships(entity1_id, entity2_id)

    def get_related_entities(self, entity_id: str) -> List[str]:
        """Get all related entities of a specified entity"""
        return self.entity.get_related_entities(entity_id)

    def merge_entities(self, entity_id1: str, entity_id2: str) -> str:
        """Merge two entities"""
        return self.entity.merge_entities(entity_id1, entity_id2)

    # Search methods
    def search_similar_entities(
        self, query_entity: str, top_n: int = 5, threshold: float = 0.8
    ) -> List[Tuple[str, float]]:
        """Search for similar entities"""
        return self.search.search_similar_entities(query_entity, top_n, threshold)

    def search_vector_store(
        self, query: str, entity_id: Optional[str] = None, k: int = 3
    ) -> List[Tuple[str, float]]:
        """Search vector store"""
        return self.search.search_vector_store(query, entity_id, k)

    def search_similar_relationships(
        self, query: str, entity_id: str, k: int = 3
    ) -> List[Tuple[str, str, str, float]]:
        """Search for similar relationships"""
        return self.search.search_similar_relationships(query, entity_id, k)

    def search_all_paths(
        self,
        start_entity: str,
        end_entity: str,
        max_depth: int = 5,
        max_results: int = 3,
    ) -> List[Dict]:
        """Search for all paths between two entities"""
        return self.search.search_all_paths(
            start_entity, end_entity, max_depth, max_results
        )

    def search_communities(
        self, query: str, top_n: int = 1
    ) -> List[Tuple[List[str], str]]:
        """Search for communities based on the query"""
        return self.search.search_communities(query, top_n)

    def tree_search(self, start_entity: str, max_depth: int = 3) -> nx.DiGraph:
        """Tree search starting from the specified entity"""
        return self.search.tree_search(start_entity, max_depth)

    # Community detection
    def detect_communities(
        self, resolution: float = 1.0, min_community_size: int = 4
    ) -> List[List[str]]:
        """Detect communities in the graph"""
        return self.entity.detect_communities(resolution, min_community_size)

    # Graph operations
    def merge_graphs(self, other_graph: "KnowledgeGraph") -> None:
        """Merge other graph into the current graph"""
        self.entity.merge_graphs(other_graph.entity)

    def merge_similar_entities(self) -> None:
        """Merge similar entities"""
        self.entity.merge_similar_entities()

    def remove_duplicates(self) -> None:
        """Remove duplicates from the graph"""
        self.entity.remove_duplicates_and_self_loops()

    # Visualization methods
    def visualize(self) -> None:
        """Create base visualization of the graph"""
        self.visualization.visualize()

    def visualize_communities(self) -> None:
        """Create visualization of the communities"""
        self.visualization.visualize_communities()

    def get_statistics(self) -> Dict[str, int]:
        """Get statistics of the graph"""
        return {
            "Entity count": self.storage.get_entity_count(),
            "Relationship count": self.storage.get_relationship_count(),
            "Alias count": self.storage.get_alias_count(),
            "Vector store count": self.storage.get_store_count(),
        }

    def generate_statistics(self, save_path: Optional[str] = None) -> str:
        """
        Generate and save the statistics of the graph

        Args:
            save_path: Optional save path. If not provided, default path will be used

        Returns:
            str: Path of the saved statistics
        """
        try:
            return self.visualization.generate_statistics(save_path)
        except Exception as e:
            print(f"[ERROR] Failed at statistics generation: {str(e)}")
            return ""

    def cleanup(self) -> None:
        """Clean up the resources"""
        self.storage.cleanup()

    def __enter__(self):
        """Entrance to the context manager"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit from the context manager, cleaning up resources"""
        self.cleanup()
