from typing import List, Tuple, Dict, Optional, Any
# from openai import OpenAI
from ollama import Client
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from graph_storage import GraphStorage
from embedding_model import EmbeddingModel
from config import *

ENTITY_MERGE_PROMPT = "prompt/entity_merge.txt"
RELATIONSHIP_MERGE_PROMPT = "prompt/relationship_merge.txt"
COMMUNITY_SUMMARY_PROMPT = "prompt/community_summary.txt"


class GraphEntity:
    """Class of GraphEntity, handles all operations related to entities and relationships"""

    def __init__(self, storage: GraphStorage, llm_client: Client):
        """
        Initialize the GraphEntity

        Args:
            storage: Store a GraphEntity instance
            llm_client: LLM client instance, used to determine whether two entities could be merged
        """
        self.storage = storage
        self.llm_client = llm_client

    def add_entity(self, entity_id: str, content_units: List[Tuple[str, str]]) -> str:
        """
        Add entity to graph, make a merge judgment if there exists two similar entities

        Args:
            entity_id: Entity ID
            content_units: The title and content list in the format of [(title, content),...]

        Returns:
            str: The main id of the entity (possibly the id after merging)
        """
        # Check whether it exists
        main_id = self._get_main_id(entity_id)
        if main_id:
            print(f"Found existing entity '{entity_id}'，now merging with main entity '{main_id}' ...")
            self._merge_entity_content(main_id, content_units)
            return main_id

        # Generate embedding of new entity
        new_embedding = EmbeddingModel.get_instance().embed_query(entity_id)

        # Check for similar entities
        for existing_id, existing_embedding in self.storage.entity_embeddings.items():
            similarity = cosine_similarity([new_embedding], [existing_embedding])[0][0]
            if similarity > 0.85:
                print(
                    f"Found high similarity entities：'{entity_id}' and '{existing_id}' have similarity of {similarity:.3f}"
                )
                should_merge = self._llm_merge_judgment(entity_id, existing_id)
                if should_merge:
                    print(
                        f"LLM decides that it is okay to merge, merging entity '{entity_id}' to entity '{existing_id}'..."
                    )
                    self._merge_entity_content(existing_id, content_units)
                    self._add_alias(existing_id, entity_id)
                    return existing_id

        # Add as a new entity
        print(f"Creating new entity：'{entity_id}'")
        self.storage.graph.add_node(entity_id)
        self.storage.entity_embeddings[entity_id] = new_embedding
        self.storage.alias_to_main_id[entity_id] = entity_id
        self.storage.save_entity(entity_id, content_units)

        return entity_id

    def add_relationship(
        self, entity1_id: str, entity2_id: str, relationship_type: str
    ) -> None:
        """
        Add a relationship between entities, merge with the most similar relationship
        When the similarity exceeds 0.95, keep the original relationship unchanged
        When the similarity is between 0.85-0.95, merge the relationship
        When the similarity is less than 0.85, add a new relationship

        Args:
            entity1_id: ID of the source entity
            entity2_id: ID of the target entity
            relationship_type: relationship type
        """
        main_id1 = self._get_main_id(entity1_id)
        main_id2 = self._get_main_id(entity2_id)

        if not (main_id1 and main_id2):
            if not main_id1:
                print(f"Entity '{entity1_id}' does not exist")
            if not main_id2:
                print(f"Entity '{entity2_id}' does not exist")
            return

        # Get the existing relationship between this pair of entities on the same direction
        existing_relationships = [
            (d["type"], k)
            for u, v, k, d in self.storage.graph.edges(data=True, keys=True)
            if u == main_id1 and v == main_id2  # only get the relationship on the same direction
        ]

        # If there is no existing relationship, add one directly
        if not existing_relationships:
            self.storage.graph.add_edge(main_id1, main_id2, type=relationship_type)
            print(f"Adding new relationship: {main_id1} -{relationship_type}-> {main_id2}")
            return

        # Get all relationships' embeddings (including the new relationship)
        new_embedding = EmbeddingModel.get_instance().embed_query(relationship_type)
        rel_embeddings = [
            (rel, key, EmbeddingModel.get_instance().embed_query(rel))
            for rel, key in existing_relationships
        ]

        # Compute the similarities with all existing relationships
        similarities = []
        for rel, key, embedding in rel_embeddings:
            similarity = cosine_similarity([new_embedding], [embedding])[0][0]
            similarities.append((similarity, rel, key))

        # Find the most similar relationship
        if similarities:
            max_similarity, most_similar_rel, edge_key = max(
                similarities, key=lambda x: x[0]
            )

            print(f"Most similar relaionship found: '{relationship_type}' and '{most_similar_rel}'")
            print(f"Similarity score is {max_similarity:.3f}")

            # If the similarity is greater than 0.95, keep the original relationship
            if max_similarity > 0.95:
                print(f"Similarity > 0.95, keeping the original relationship：'{most_similar_rel}'")
                return

            # If the similarity is between 0.85-0.95, merge the relationship
            elif max_similarity > 0.85:
                merged_relation = self._llm_merge_relationships(
                    main_id1, main_id2, relationship_type, most_similar_rel
                )
                print(f"Merging relationship: '{merged_relation}'")

                # Update the relationship
                self.storage.graph.remove_edge(main_id1, main_id2, edge_key)
                self.storage.graph.add_edge(main_id1, main_id2, type=merged_relation)
                return

        # If there is no similar relationship or the similarity is no greater than 0.85, add new relationship
        self.storage.graph.add_edge(main_id1, main_id2, type=relationship_type)
        print(f"Adding new relationship: {main_id1} -{relationship_type}-> {main_id2}")

    def get_entity_info(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get entity information

        Args:
            entity_id: Entity ID

        Returns:
            Optional[Dict[str, Any]]: Dictionary of entity information, including main ID, content, and alias
        """
        main_id = self._get_main_id(entity_id)
        if not main_id:
            return None

        content = self.storage.load_entity(main_id)
        aliases = list(self.storage.entity_aliases.get(main_id, []))

        return {"main_id": main_id, "content": content, "aliases": aliases}

    def get_relationships(self, entity1_id: str, entity2_id: str) -> List[str]:
        """
        Get all relationships between two entities

        Args:
            entity1_id: ID of the 1st entity
            entity2_id: ID of the 2nd entity

        Returns:
            List[str]: list of relationship types
        """
        main_id1 = self._get_main_id(entity1_id)
        main_id2 = self._get_main_id(entity2_id)

        if main_id1 and main_id2:
            return [
                d["type"]
                for u, v, d in self.storage.graph.edges(data=True)
                if u == main_id1 and v == main_id2
            ]
        return []

    def get_related_entities(self, entity_id: str) -> List[str]:
        """
        Get all entities related to the specified entity

        Args:
            entity_id: Entity ID

        Returns:
            List[str]: ID list of related entities
        """
        main_id = self._get_main_id(entity_id)
        if main_id:
            successors = list(self.storage.graph.successors(main_id))
            predecessors = list(self.storage.graph.predecessors(main_id))
            return list(set(successors + predecessors))
        return []

    def merge_entities(self, entity_id1: str, entity_id2: str) -> str:
        """
        Manually merge two entities

        Args:
            entity_id1: ID of the 1st entity
            entity_id2: ID of the 2nd entity

        Returns:
            str: ID of the main entity after merging
        """
        main_id1 = self._get_main_id(entity_id1)
        main_id2 = self._get_main_id(entity_id2)

        if not (main_id1 and main_id2):
            if not main_id1:
                print(f"Entity '{entity_id1}' does not exist")
            if not main_id2:
                print(f"Entity '{entity_id2}' does not exist")
            return ""

        # Keep the entity with the shorter ID as the main entity
        main_entity = main_id1 if len(main_id1) <= len(main_id2) else main_id2
        merged_entity = main_id2 if main_entity == main_id1 else main_id1

        # Merge content
        merged_content = self.storage.load_entity(merged_entity)
        self._merge_entity_content(main_entity, merged_content)

        # Merge relationships
        self._merge_entity_relationships(main_entity, merged_entity)

        # Merge aliases
        self._merge_entity_aliases(main_entity, merged_entity)

        # Delete merged entities
        self._remove_entity(merged_entity)

        return main_entity

    def merge_similar_entities(self) -> None:
        """Automatically check and merge similar entities"""
        print("\nStart checking and merging similar entities...")

        # Get all similarities of entity pairs
        entity_pairs = []
        entities = list(self.storage.graph.nodes())

        for i, entity1 in enumerate(entities):
            embedding1 = self.storage.entity_embeddings[entity1]
            for entity2 in entities[i + 1 :]:
                embedding2 = self.storage.entity_embeddings[entity2]
                similarity = cosine_similarity([embedding1], [embedding2])[0][0]
                if similarity > 0.85:
                    if self._llm_merge_judgment(entity1, entity2):
                        entity_pairs.append((entity1, entity2, similarity))

        # Sort by similarity
        entity_pairs.sort(key=lambda x: x[2], reverse=True)

        # Excetute merging
        merged_entities = set()
        for entity1, entity2, similarity in entity_pairs:
            if entity1 not in merged_entities and entity2 not in merged_entities:
                print(f"\nMerging entities {entity1} and {entity2} (Similarity: {similarity:.3f})")
                merged_id = self.merge_entities(entity1, entity2)
                merged_entities.add(entity2 if merged_id == entity1 else entity1)

    def _get_main_id(self, entity_id: str) -> Optional[str]:
        """Get the main ID of the entity"""
        if entity_id in self.storage.alias_to_main_id:
            return self.storage.alias_to_main_id[entity_id]
        if entity_id in self.storage.graph.nodes():
            return entity_id
        return None

    def _merge_entity_content(
        self, main_id: str, content_units: List[Tuple[str, str]]
    ) -> None:
        """Merge the content of the entity"""
        existing_content = self.storage.load_entity(main_id)

        # Deduplicate with union operation
        existing_set = {
            (title.strip(), content.strip()) for title, content in existing_content
        }
        new_set = {(title.strip(), content.strip()) for title, content in content_units}
        merged_set = existing_set.union(new_set)

        # Save the merged content
        self.storage.save_entity(main_id, list(merged_set))

    def _merge_entity_relationships(self, main_id: str, merged_id: str) -> None:
        """Merge relationships of the entity"""
        # Handle in-edges
        for predecessor in self.storage.graph.predecessors(merged_id):
            if predecessor != main_id:  # Prevent self-loop
                edges_data = self.storage.graph.get_edge_data(predecessor, merged_id)
                for edge_data in edges_data.values():
                    # Prevent self-loop
                    if predecessor != main_id:
                        self.add_relationship(predecessor, main_id, edge_data["type"])

        # Handle out-edges
        for successor in self.storage.graph.successors(merged_id):
            if successor != main_id:  # Prevent self-loop
                edges_data = self.storage.graph.get_edge_data(merged_id, successor)
                for edge_data in edges_data.values():
                    # Prevent self-loop
                    if successor != main_id:
                        self.add_relationship(main_id, successor, edge_data["type"])

    def _merge_entity_aliases(self, main_id: str, merged_id: str) -> None:
        """Merge aliases of the entity"""
        if merged_id in self.storage.entity_aliases:
            for alias in self.storage.entity_aliases[merged_id]:
                self._add_alias(main_id, alias)
            del self.storage.entity_aliases[merged_id]
        self._add_alias(main_id, merged_id)

    def _add_alias(self, main_id: str, alias: str) -> None:
        """Add an alias to the entity"""
        if main_id not in self.storage.entity_aliases:
            self.storage.entity_aliases[main_id] = set()
        self.storage.entity_aliases[main_id].add(alias)
        self.storage.alias_to_main_id[alias] = main_id

    def _remove_entity(self, entity_id: str) -> None:
        """Delete the entity from the graph"""
        self.storage.graph.remove_node(entity_id)
        if entity_id in self.storage.entity_embeddings:
            del self.storage.entity_embeddings[entity_id]

    def _llm_merge_relationships(
        self, entity1: str, entity2: str, rel1: str, rel2: str
    ) -> str:
        """
        Use LLM to merge two relationship descriptions, considering the entity context

        Args:
            entity1: source entity
            entity2: target entity
            rel1: relationship type 1
            rel2: relationship type 2

        Returns:
            str: The description of the merged relationship
        """
        try:
            with open(RELATIONSHIP_MERGE_PROMPT, "r", encoding="utf-8") as file:
                template = file.read()

            prompt = template.format(
                entity1=entity1, entity2=entity2, rel1=rel1, rel2=rel2
            )

            messages = [{"role": "user", "content": prompt}]

            # response = self.llm_client.chat.completions.create(
            #     model="moonshot-v1-8k", messages=messages, temperature=0.5
            # )

            response = self.llm_client.chat(model=LLM_MODEL_NAME, messages=messages)

            # return response.choices[0].message.content.strip()
            return response.message.content.strip()

        except Exception as e:
            print(f"LLm failed to merge relationships: {str(e)}")
            return rel1  # Keep the first relationship when an error occurs

    def _llm_merge_judgment(self, entity1: str, entity2: str) -> bool:
        """Use LLM to determine whether two entities should be merged"""
        try:
            with open(ENTITY_MERGE_PROMPT, "r", encoding="utf-8") as file:
                template = file.read()

            prompt = template.format(entity1=entity1, entity2=entity2)
            messages = [{"role": "user", "content": prompt}]

            # response = self.llm_client.chat.completions.create(
            #     model="moonshot-v1-8k", messages=messages, temperature=0.5
            # )
            response = self.llm_client.chat(model=LLM_MODEL_NAME, messages=messages)

            # result = response.choices[0].message.content.strip().lower()
            result = response.message.content.strip().lower()
            return result == "yes"

        except Exception as e:
            print(f"LLM judgment error: {str(e)}")
            return False

    def remove_duplicates_and_self_loops(self) -> None:
        """Remove duplicated edges and self-loop (alias included)"""
        changes_made = False  # Keep track of any changes made

        # Remove direct self-loops
        for u, v, data in list(nx.selfloop_edges(self.storage.graph, data=True)):
            print(f"Remove self-loop edges: {u} -> {v}, type of relationship: {data.get('type')}")
            self.storage.graph.remove_edge(u, v)
            changes_made = True

        # Remove self-loops caused by aliases
        for source, target, data in list(self.storage.graph.edges(data=True)):
            source_main = self.storage.alias_to_main_id.get(source, source)
            target_main = self.storage.alias_to_main_id.get(target, target)

            if source_main == target_main:
                print(
                    f"Remove alias self-loop edges: {source} -> {target}, type of relationship: {data.get('type')}, main entity: {source_main}"
                )
                self.storage.graph.remove_edge(source, target)
                changes_made = True

        # Remove duplicated edges
        edges_to_remove = []
        for u, v, keys, data in self.storage.graph.edges(keys=True, data=True):
            edge_type = data.get("type")
            edge_data = self.storage.graph.get_edge_data(u, v)

            if edge_data:
                existing_edges = [
                    (k, d)
                    for k, d in edge_data.items()
                    if k != keys and d.get("type") == edge_type
                ]
                for k, _ in existing_edges:
                    edges_to_remove.append((u, v, k))
                    print(f"Remove duplicated edges: {u} -> {v}, type of relationship: {edge_type}")

        for edge in edges_to_remove:
            self.storage.graph.remove_edge(*edge)
            changes_made = True

        # If there are any changes, save the updated graph
        if changes_made:
            print("Changes in graph detected, saving the updated graph...")
            self.storage.save()
            print("The graph has been updated and saved!")
        else:
            print("No duplicated edges or self-loops detected, no need to update")

    def merge_graphs(self, other_entity: "GraphEntity") -> None:
        """
        Merge the entities and relationships from another graph to the current graph

        Args:
            other_entity: The instance of the entity manager of the graph to be merged
        """
        print("Initialize merging graphs...")

        # 1. Merge nodes and content
        for node in other_entity.storage.graph.nodes():
            print(f"\Processing node: {node}")
            node_info = other_entity.get_entity_info(node)
            if node_info:
                # Check if the node already exists
                main_id = self._get_main_id(node)
                if main_id:
                    print(f"Node '{node}' exists, the main ID is '{main_id}'")
                    # Merge entity content
                    self._merge_entity_content(main_id, node_info["content"])

                    # Merge aliases
                    for alias in node_info["aliases"]:
                        if alias not in self.storage.alias_to_main_id:
                            self._add_alias(main_id, alias)
                            print(f"Add alias: {alias} -> {main_id}")
                else:
                    # Add new node
                    print(f"添加新节点: {node}")
                    new_id = self.add_entity(node, node_info["content"])
                    # Add aliases
                    for alias in node_info["aliases"]:
                        self._add_alias(new_id, alias)

        # 2. Merge relationships
        for edge in other_entity.storage.graph.edges(data=True):
            source, target, data = edge
            source_main = self._get_main_id(source)
            target_main = self._get_main_id(target)

            if source_main and target_main:
                # Check if the relationship already exists
                existing_relationships = self.get_relationships(
                    source_main, target_main
                )
                if data["type"] not in existing_relationships:
                    self.add_relationship(source_main, target_main, data["type"])
                    print(f"Add new relationship: {source_main} -{data['type']}-> {target_main}")
                else:
                    print(f"The relationship exists: {source_main} -{data['type']}-> {target_main}")

        # 3. Rebuid the vector store
        print("\nUpdating vector store...")
        # Update entity vector store
        for node in self.storage.graph.nodes():
            content = self.storage.load_entity(node)
            if content:
                self.storage._create_entity_vector_store(node, content)
                print(f"Update on entity '{node}' vector store completed")

        # 4. Save updated graph
        self.storage.save()
        print("\nKnowledge graphs merged!")

        # 5. Print summary stats after merging
        print("\nSummary stats:")
        print(f"- Total nodes: {len(self.storage.graph.nodes())}")
        print(f"- Total edges: {len(self.storage.graph.edges())}")
        print(
            f"- Total aliases: {sum(len(aliases) for aliases in self.storage.entity_aliases.values())}"
        )
        print(f"- Total vector stores: {len(self.storage.vector_stores)}")

    def detect_communities(
        self, resolution: float = 1.2, min_community_size: int = 4
    ) -> Dict[int, Dict]:
        """
        Detect and analyze communities in the graph

        Args:
            resolution: The resolution parameter to detect a community
            min_community_size: Minimum community size

        Returns:
            Dict[int, Dict]: Dictionary of community information
        """
        # Get the copy of the graph and remove self-loops
        G = self.storage.graph.copy()
        G.remove_edges_from(nx.selfloop_edges(G))
        print("Community inspection initiated：number of nodes:", len(G.nodes), "number of edges:", len(G.edges))

        # Use Louvain method to detect communities
        raw_communities = nx.community.louvain_communities(
            G, resolution=resolution, seed=42
        )
        print("The number of communities detected:", len(raw_communities))

        # Analyze each community
        communities_data = {}
        for idx, members in enumerate(raw_communities):
            if len(members) < min_community_size:
                print(
                    f"Skip community {idx} since its member number {len(members)} is lower than threshold {min_community_size}"
                )
                continue

            print(f"\Processing community {idx}，member count: {len(members)}")
            members_list = list(members)

            # Get central members
            central_members = self._identify_central_members(members_list)
            print(f"The central member of community {idx} is: {central_members}")

            # Get all relationships in the community
            community_relations = self._get_community_relations(members_list)
            print(f"The number of relationships of community {idx} is : {len(community_relations)}")

            # Generate community summary
            summary = self._generate_community_summary(
                members_list, central_members, community_relations
            )
            print(f"The community summary of community {idx} is: {summary[:200]}...")  # only print the first 200 characters

            # Create community information dictionary
            communities_data[idx] = {
                "members": members_list,
                "central_members": central_members,
                "relations": community_relations,
                "summary": summary,
            }

        # Save community data and summary
        self.storage.save_communities(communities_data)
        self.storage.save_community_summaries(communities_data)
        print("\nCommunity inspection completed, results have been saved.")

        return communities_data

    def _identify_central_members(self, members: List[str]) -> List[str]:
        """
        Identify the central members of the community

        Args:
            members: a list of community members

        Returns:
            List[str]: a list of central community members
        """
        # Calculate the degree of each member
        member_degrees = {}
        for member in members:
            successors = set(self.storage.graph.successors(member))
            predecessors = set(self.storage.graph.predecessors(member))
            # Only consider connections within the community
            community_connections = len(
                [n for n in successors.union(predecessors) if n in members]
            )
            member_degrees[member] = community_connections

        # Select the top 4 members with the highest degree
        central_members = sorted(
            member_degrees.items(), key=lambda x: x[1], reverse=True
        )[:4]

        return [member for member, _ in central_members]

    def _get_community_relations(self, members: List[str]) -> List[Dict]:
        """
        Get all relationships within the community

        Args:
            members: a list of community members

        Returns:
            List[Dict]: List of relationships, where each relationship includes source, target, type
        """
        relations = []
        for source in members:
            for target in self.storage.graph.successors(source):
                if target in members:
                    edges = self.storage.graph.get_edge_data(source, target)
                    for edge_data in edges.values():
                        relations.append(
                            {
                                "source": source,
                                "target": target,
                                "type": edge_data["type"],
                            }
                        )
        return relations

    def _generate_community_summary(
        self, members: List[str], central_members: List[str], relations: List[Dict]
    ) -> str:
        """
        Generate community summary

        Args:
            members: a list of all community members
            central_members: a list of central members
            relations: a list of relationships within the community

        Returns:
            str: a community summary
        """
        try:
            # 1. Format the central members information
            core_entities_info = [f"- {entity}" for entity in central_members]

            # 2. Process relationships
            # 2.1 Calculate the connections of entities
            entity_connections = {member: 0 for member in members}
            for rel in relations:
                entity_connections[rel["source"]] = (
                    entity_connections.get(rel["source"], 0) + 1
                )
                entity_connections[rel["target"]] = (
                    entity_connections.get(rel["target"], 0) + 1
                )

            # 2.2 Categorize by relationship tye and calculate the weights respectively
            relation_groups = {}
            for rel in relations:
                rel_type = rel["type"]
                if rel_type not in relation_groups:
                    relation_groups[rel_type] = []

                # Calculate the weight of the relationship (the total connection degree of the two entities)
                weight = (
                    entity_connections[rel["source"]]
                    + entity_connections[rel["target"]]
                )
                relation_groups[rel_type].append(
                    {
                        "source": rel["source"],
                        "target": rel["target"],
                        "weight": weight,
                        "type": rel_type,
                    }
                )

            # 2.3 Rank the entity by weight in each relationship type
            relation_info = []
            for rel_type, rel_list in relation_groups.items():
                # Rank the entity pairs by weight
                sorted_rels = sorted(rel_list, key=lambda x: x["weight"], reverse=True)
                # Format the relationship information
                examples = [
                    f"{rel['source']}-{rel['type']}-{rel['target']}"
                    for rel in sorted_rels[:3]
                ]  # Only take the top 3 with the highest weight
                relation_info.append(f"- {'; '.join(examples)}")

            # 3. Load and fill the template
            with open(COMMUNITY_SUMMARY_PROMPT, "r", encoding="utf-8") as f:
                prompt_template = f.read()

            # 4. Fill the template
            prompt = prompt_template.format(
                core_entities="\n".join(core_entities_info),
                relationships="\n".join(relation_info),
            )

            # 5. Generate the community summary
            response = self.llm_client.chat.completions.create(
                model="moonshot-v1-auto",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
            )

            return response.choices[0].message.content.strip().replace("\n", " ")

        except Exception as e:
            print(f"[ERROR] Error while generating community summary: {str(e)}")
