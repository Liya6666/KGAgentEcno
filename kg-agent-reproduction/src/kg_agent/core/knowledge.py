import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
import json
import asyncio
from datetime import datetime

class KnowledgeGraphInterface:
    """知识图谱接口"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.graph = nx.MultiDiGraph()
        self.entity_embeddings = {}
        self.relation_embeddings = {}
        
        # 初始化基础知识图谱
        self._initialize_base_kg()
    
    def _initialize_base_kg(self):
        """初始化基础知识图谱"""
        # 添加基础实体类型
        entity_types = [
            'person', 'organization', 'location', 'event', 'concept', 'product'
        ]
        
        for entity_type in entity_types:
            self.graph.add_node(f"entity_type_{entity_type}", 
                              type='meta', 
                              category='entity_type')
        
        # 添加基础关系类型
        relation_types = [
            'is_a', 'part_of', 'located_in', 'works_for', 'created_by', 'related_to'
        ]
        
        for relation_type in relation_types:
            self.graph.add_node(f"relation_type_{relation_type}", 
                              type='meta', 
                              category='relation_type')
    
    async def query_subgraph(self, entities: List[str], 
                           relations: List[str], 
                           max_depth: int = 3) -> Dict[str, Any]:
        """查询子图"""
        # 构建查询子图
        subgraph = {
            'entities': {},
            'relations': [],
            'paths': []
        }
        
        # 查找相关实体
        for entity in entities:
            if self.graph.has_node(entity):
                subgraph['entities'][entity] = dict(self.graph.nodes[entity])
        
        # 查找相关关系
        for entity in entities:
            if self.graph.has_node(entity):
                # 查找直接连接
                neighbors = list(self.graph.neighbors(entity))
                for neighbor in neighbors:
                    if neighbor in entities:
                        edge_data = self.graph.get_edge_data(entity, neighbor)
                        if edge_data:
                            for key, data in edge_data.items():
                                subgraph['relations'].append({
                                    'source': entity,
                                    'target': neighbor,
                                    'relation': data.get('relation', 'unknown'),
                                    'weight': data.get('weight', 1.0)
                                })
        
        # 查找路径
        if len(entities) >= 2:
            for i in range(len(entities)):
                for j in range(i+1, len(entities)):
                    try:
                        paths = list(nx.all_simple_paths(
                            self.graph, 
                            entities[i], 
                            entities[j], 
                            cutoff=max_depth
                        ))
                        subgraph['paths'].extend(paths)
                    except nx.NetworkXNoPath:
                        pass
        
        return subgraph
    
    async def add_entity(self, entity_id: str, entity_data: Dict[str, Any]):
        """添加实体"""
        self.graph.add_node(entity_id, **entity_data)
    
    async def add_relation(self, source: str, target: str, 
                          relation: str, properties: Dict[str, Any] = None):
        """添加关系"""
        properties = properties or {}
        self.graph.add_edge(source, target, relation=relation, **properties)
    
    async def search_entities(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """搜索实体"""
        results = []
        
        for node_id, node_data in self.graph.nodes(data=True):
            match = True
            for key, value in query.items():
                if key not in node_data or node_data[key] != value:
                    match = False
                    break
            
            if match:
                results.append({
                    'id': node_id,
                    'properties': dict(node_data)
                })
        
        return results
    
    async def get_entity_embedding(self, entity_id: str) -> Optional[List[float]]:
        """获取实体嵌入"""
        return self.entity_embeddings.get(entity_id)
    
    async def get_relation_embedding(self, relation: str) -> Optional[List[float]]:
        """获取关系嵌入"""
        return self.relation_embeddings.get(relation)
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """获取图谱统计"""
        return {
            'num_entities': self.graph.number_of_nodes(),
            'num_relations': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'connected_components': nx.number_weakly_connected_components(self.graph)
        }
    
    def export_graph(self, format_type: str = 'json') -> str:
        """导出图谱"""
        if format_type == 'json':
            data = {
                'nodes': [dict(node) for node in self.graph.nodes(data=True)],
                'edges': [dict(edge) for edge in self.graph.edges(data=True)]
            }
            return json.dumps(data, indent=2, default=str)
        
        return ""