import os
import json
import base64
import shutil
from typing import List, Tuple, Dict, Optional, Set, Any
import networkx as nx
import numpy as np
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from embedding_model import EmbeddingModel


class GraphStorage:
    """GraphStorage manager, handles all storage-related operations"""

    def __init__(self, base_path: str):
        """
        Initialize the GraphStorage manager

        Args:
            base_path: Base path for the knowledge graph data
        """
        # Base path
        self.base_path = base_path

        # Core file paths (directly under base_path)
        self.graph_file = os.path.join(base_path, "graph.json")  # Graph file
        self.embeddings_file = os.path.join(
            base_path, "embeddings.json"
        )  # Entity embeddings file
        self.global_doc_path = os.path.join(base_path, "global.md")  # Global document

        # Child file paths
        self.entity_path = os.path.join(base_path, "entities")  # Entity file directory
        self.vector_path = os.path.join(base_path, "vectors")  # Vector store directory

        # Core components
        self.graph = nx.MultiDiGraph()

        # Entity management
        self.entity_embeddings: Dict[str, np.ndarray] = {}  # Entity embeddings
        self.entity_aliases: Dict[str, Set[str]] = {}  # Entity aliases
        self.alias_to_main_id: Dict[str, str] = {}  # Maps from alias to main entity

        # Vector stores
        self.vector_stores: Dict[str, FAISS] = {}  # Entity vector stores
        self.global_vector_store: Optional[FAISS] = None  # Global vector store
        self.global_content: Set[str] = set()  # Global document content

        # Add storage paths for community-related data
        self.community_file = os.path.join(base_path, "communities.json")
        self.community_summary_path = os.path.join(base_path, "community_summaries.md")
        self.communities: Dict[int, Dict] = {}  # {community_id: community_data}
        self.community_vector_store: Optional[FAISS] = None

        # Modified entities tracking: only tracks entity modifications
        self.modified_entities: Set[str] = set()

    def _init_storage(self) -> None:
        """Initialize the storage structure"""
        # Create necessary directories and files
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(self.entity_path, exist_ok=True)
        os.makedirs(self.vector_path, exist_ok=True)

        # Create global document if it doesn't exist
        if not os.path.exists(self.global_doc_path):
            with open(self.global_doc_path, "w", encoding="utf-8") as f:
                pass

    def save(self) -> None:
        """Save the knowledge graph data"""
        # Save graph structure and aliases
        graph_data = {
            "graph": nx.node_link_data(self.graph, edges="links"),
            "aliases": {k: list(v) for k, v in self.entity_aliases.items()},
            "alias_to_main_id": self.alias_to_main_id,
        }
        with open(self.graph_file, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)

        # Save entity embeddings
        embeddings_data = {}
        for k, v in self.entity_embeddings.items():
            if not isinstance(v, np.ndarray):
                v = np.array(v)
            embeddings_data[k] = v.tolist()
        with open(self.embeddings_file, "w", encoding="utf-8") as f:
            json.dump(embeddings_data, f)

        # Update the vector stores of modified entities
        for entity_id in self.modified_entities:
            if entity_id in self.graph.nodes():
                content = self.load_entity(entity_id)
                if content:
                    self._create_entity_vector_store(entity_id, content)
                    print(f"Update the vector store of entity '{entity_id}'")

        # Update the global vector store
        if os.path.exists(self.global_doc_path):
            self._create_global_vector_store()
            print(f"Update global vector store")

        # Clear the modified entities tracking
        self.modified_entities.clear()

    def load(self) -> None:
        """Load the knowledge graph data"""
        # Load graph structure and aliases
        if os.path.exists(self.graph_file):
            print(f"Existing knowledge graph detected at '{self.base_path}', loading...")
            with open(self.graph_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.graph = nx.node_link_graph(data["graph"], multigraph=True)
            self.entity_aliases = {k: set(v) for k, v in data["aliases"].items()}
            self.alias_to_main_id = data["alias_to_main_id"]

            # Loading entity embeddings
            if os.path.exists(self.embeddings_file):
                print("Loading entity embedding...")
                with open(self.embeddings_file, "r", encoding="utf-8") as f:
                    embeddings_data = json.load(f)
                # Convert list to numpy array directly, as the loaded data is already in list form
                self.entity_embeddings = {
                    k: np.array(v) for k, v in embeddings_data.items()
                }
            else:
                print("Didn't find entity embedding, now regenerating...")
                self._regenerate_embeddings()

            # Load global document content
            if os.path.exists(self.global_doc_path):
                with open(self.global_doc_path, "r", encoding="utf-8") as f:
                    self.global_content = set(f.read().split("\n\n"))

            # Load vector stores
            self._load_vector_stores()

            # Load community data
            self._load_community_data()

    def _load_community_data(self) -> None:
        """Try to load community-related data"""
        # Check and load community JSON data
        if os.path.exists(self.community_file):
            print("Community data detected, loading...")
            try:
                with open(self.community_file, "r", encoding="utf-8") as f:
                    self.communities = json.load(f)
                print(f"{len(self.communities)} community data have been loaded")

                # Load community vector store
                store_path = os.path.join(self.vector_path, "community_summaries")
                if os.path.exists(store_path):
                    print("Loading community summary vector store...")
                    self.community_vector_store = FAISS.load_local(
                        store_path,
                        EmbeddingModel.get_instance(),
                        allow_dangerous_deserialization=True,
                    )
                    print("Community summary vector store successfully loaded")
                else:
                    print("Creating vector store for community summary...")
                    self._create_community_summary_store()

            except Exception as e:
                print(f"[ERROR] Loading community data: {str(e)}")
                self.communities = {}
                self.community_vector_store = None
        else:
            print("Didn't find community data, skip loading")
            self.communities = {}
            self.community_vector_store = None

    def save_entity(self, entity_id: str, content_units: List[Tuple[str, str]]) -> None:
        """
        Save entity data

        Args:
            entity_id: Entity ID
            content_units: Content unit list in the format of [(title, content),...]
        """
        # Save to markdown file
        file_path = os.path.join(self.entity_path, f"{entity_id}.md")
        with open(file_path, "w", encoding="utf-8") as f:
            for title, content in content_units:
                f.write(f"# {title}\n\n{content}\n\n")

        # Update global document
        self._update_global_document(content_units)

        # Mark the entity as modified
        self.modified_entities.add(entity_id)

    def load_entity(self, entity_id: str) -> List[Tuple[str, str]]:
        """
        Load entity data

        Args:
            entity_id: Entity ID

        Returns:
            List[Tuple[str, str]]: Content unit list
        """
        file_path = os.path.join(self.entity_path, f"{entity_id}.md")
        if not os.path.exists(file_path):
            return []

        content_units = []
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            title = ""
            content = ""
            for line in lines:
                if line.startswith("# "):
                    if title and content:
                        content_units.append((title.strip(), content.strip()))
                        content = ""
                    title = line[2:].strip()
                else:
                    content += line
            if title and content:
                content_units.append((title.strip(), content.strip()))
        return content_units

    def save_communities(self, communities_data: Dict[int, Dict]) -> None:
        """Save community data to JSON"""
        self.communities = communities_data
        with open(self.community_file, "w", encoding="utf-8") as f:
            json.dump(communities_data, f, ensure_ascii=False, indent=2)

    def save_community_summaries(self, communities_data: Dict[int, Dict]) -> None:
        """Generate and save community summary document"""
        # Generated markdown format community summary
        doc_content = []
        for comm_id, comm_data in communities_data.items():
            doc_content.append(f"# Community_{comm_id}\n")
            doc_content.append(f"{comm_data['summary']}\n\n")

        # Save the summary document
        with open(self.community_summary_path, "w", encoding="utf-8") as f:
            f.write("".join(doc_content))

        # Create vector store
        self._create_community_summary_store()

    def get_entity_count(self) -> int:
        """Get the number of entities"""
        return len(self.graph.nodes())

    def get_relationship_count(self) -> int:
        """Get the number of relationships"""
        return len(self.graph.edges())

    def get_alias_count(self) -> int:
        """Get the number of aliases"""
        return sum(len(aliases) for aliases in self.entity_aliases.values())

    def get_store_count(self) -> int:
        """Get the number of vector stores"""
        return len(self.vector_stores)

    def _regenerate_embeddings(self) -> None:
        """Regenerate entity embeddings"""
        self.entity_embeddings = {}
        for node in self.graph.nodes():
            embedding = EmbeddingModel.get_instance().embed_query(node)
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            self.entity_embeddings[node] = embedding

    def _load_vector_stores(self) -> None:
        """Load vector stores"""
        # Load global vector store
        global_store_path = os.path.join(self.vector_path, "global")
        if os.path.exists(global_store_path):
            self.global_vector_store = FAISS.load_local(
                global_store_path,
                EmbeddingModel.get_instance(),
                allow_dangerous_deserialization=True,
            )
        elif os.path.exists(self.global_doc_path):
            self._create_global_vector_store()

        # Load entity vector stores
        for node in self.graph.nodes():
            store_path = os.path.join(self.vector_path, self._encode_filename(node))
            if os.path.exists(store_path):
                try:
                    vector_store = FAISS.load_local(
                        store_path,
                        EmbeddingModel.get_instance(),
                        allow_dangerous_deserialization=True,
                    )
                    self.vector_stores[node] = vector_store
                except Exception as e:
                    print(f"Failed to load the vector store of entity '{node}': {str(e)}")
                    content = self.load_entity(node)
                    if content:
                        self._create_entity_vector_store(node, content)
            else:
                print(f"The vector store of entity '{node}' does not exist, now generating...")
                content = self.load_entity(node)
                if content:
                    self._create_entity_vector_store(node, content)

    def _create_community_summary_store(self) -> None:
        """Create a vector store for community summaries"""
        if not os.path.exists(self.community_summary_path):
            print("Creation failed, no community summary")
            return

        store_path = os.path.join(self.vector_path, "community_summaries")

        # Use headers to split the document
        headers_to_split_on = [("#", "Community")]
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

        with open(self.community_summary_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split the document
        docs = splitter.split_text(content)

        # Create vector store
        self.community_vector_store = FAISS.from_documents(
            documents=docs,
            embedding=EmbeddingModel.get_instance(),
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        )

        # Save vector store
        self.community_vector_store.save_local(store_path)
        print("Community summary vector store creation complete")

    def _create_entity_vector_store(
        self, entity_id: str, content_units: List[Tuple[str, str]]
    ) -> None:
        """
        Create a vector store for the entity

        Args:
            entity_id: Entity ID
            content_units: Content unit list
        """
        store_path = os.path.join(self.vector_path, self._encode_filename(entity_id))

        # Construct markdown text
        markdown_text = ""
        for title, content in content_units:
            markdown_text += f"# {title}\n\n{content}\n\n"

        # Split the document
        headers_to_split_on = [("#", "Header 1")]
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
        docs = md_splitter.split_text(markdown_text)

        # Create vector store
        vector_store = FAISS.from_documents(
            documents=docs,
            embedding=EmbeddingModel.get_instance(),
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        )

        # Save
        vector_store.save_local(store_path)
        self.vector_stores[entity_id] = vector_store

    def _create_global_vector_store(self) -> None:
        """Create a global vector store"""
        if not os.path.exists(self.global_doc_path):
            return

        store_path = os.path.join(self.vector_path, "global")

        # Load and split the document
        with open(self.global_doc_path, "r", encoding="utf-8") as f:
            content = f.read()

        headers_to_split_on = [("#", "Header 1")]
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
        docs = md_splitter.split_text(content)

        # Create vector store
        self.global_vector_store = FAISS.from_documents(
            documents=docs,
            embedding=EmbeddingModel.get_instance(),
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        )

        # Save
        self.global_vector_store.save_local(store_path)

    def _update_global_document(self, content_units: List[Tuple[str, str]]) -> None:
        """
        Update the global document

        Args:
            content_units: New content units
        """
        new_content = set()
        for title, content in content_units:
            new_content.add(f"# {title}\n\n{content}\n\n")

        new_entries = new_content - self.global_content
        if new_entries:
            with open(self.global_doc_path, "a", encoding="utf-8") as f:
                for entry in new_entries:
                    f.write(entry)
            self.global_content.update(new_entries)

    @staticmethod
    def _encode_filename(filename: str) -> str:
        """文件名编码"""
        filename_bytes = filename.encode("utf-8")
        encoded_bytes = base64.urlsafe_b64encode(filename_bytes)
        return encoded_bytes.decode("utf-8")

    @staticmethod
    def _decode_filename(encoded_filename: str) -> str:
        """Decode the encoded filename"""
        try:
            decoded_bytes = base64.urlsafe_b64decode(encoded_filename.encode("utf-8"))
            return decoded_bytes.decode("utf-8")
        except Exception as e:
            print(f"[ERROR] Decoding error: {str(e)}")
            return encoded_filename

    def cleanup(self) -> None:
        """
        Cleanup resources
        Mainly clean up cached data in memory
        """
        try:
            # Save current state
            self.save()

            # Clean up vector store references in memory
            self.vector_stores.clear()
            self.global_vector_store = None
            self.community_vector_store = None

            # Clean up other memory caches
            self.entity_embeddings.clear()
            self.global_content.clear()
            self.modified_entities.clear()
            self.communities.clear()

        except Exception as e:
            print(f"[ERROR] Cleanup error: {str(e)}")

    def remove_entity(self, entity_id: str) -> None:
        """
        Delete entity and related data

        Args:
            entity_id: Entity ID
        """
        try:
            # Remove from the modified tracking
            self.modified_entities.discard(entity_id)

            # Remove entity document
            file_path = os.path.join(self.entity_path, f"{entity_id}.md")
            if os.path.exists(file_path):
                os.remove(file_path)

            # Remove vector store
            store_path = os.path.join(
                self.vector_path, self._encode_filename(entity_id)
            )
            if os.path.exists(store_path):
                shutil.rmtree(store_path)  # Directly remove the vector store directory
                if entity_id in self.vector_stores:
                    del self.vector_stores[entity_id]  # Remove reference from memory

            # Remove entity embeddings
            if entity_id in self.entity_embeddings:
                del self.entity_embeddings[entity_id]

            # Remove the node from the graph (this will automatically remove related edges)
            if entity_id in self.graph:
                self.graph.remove_node(entity_id)

            # Update aliases
            if entity_id in self.entity_aliases:
                aliases = self.entity_aliases[entity_id]
                for alias in aliases:
                    if alias in self.alias_to_main_id:
                        del self.alias_to_main_id[alias]
                del self.entity_aliases[entity_id]

            print(f"Successfully deleted entity '{entity_id}' and related data")

        except Exception as e:
            print(f"[ERROR] Error while deleting entity '{entity_id}': {str(e)}")

    def __enter__(self):
        """Entrance to the context manager"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit to the context manager, making sure resources are released properly"""
        self.cleanup()
