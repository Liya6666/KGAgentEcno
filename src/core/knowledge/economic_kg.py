import networkx as nx
from typing import Dict, List, Any, Tuple
import json
from datetime import datetime

class EconomicKnowledgeGraph:
    """
    经济知识图谱
    整合经济理论、市场数据、智能体行为等多维知识
    """
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.entity_types = {
            'agent': '经济智能体',
            'market': '市场',
            'commodity': '商品',
            'policy': '政策',
            'event': '经济事件',
            'indicator': '经济指标'
        }
        self._initialize_base_knowledge()
    
    def _initialize_base_knowledge(self):
        """初始化基础经济知识"""
        # 添加基础经济概念
        base_entities = [
            ('GDP', 'indicator', {'type': 'macro', 'unit': 'currency'}),
            ('inflation', 'indicator', {'type': 'macro', 'unit': 'percentage'}),
            ('unemployment', 'indicator', {'type': 'macro', 'unit': 'percentage'}),
            ('supply_demand', 'market', {'type': 'fundamental'}),
            ('price_elasticity', 'commodity', {'type': 'theory'}),
            ('risk_preference', 'agent', {'type': 'behavioral'})
        ]
        
        for entity_id, entity_type, attrs in base_entities:
            self.graph.add_node(entity_id, 
                              type=entity_type, 
                              attributes=attrs,
                              created_at=datetime.now().isoformat())
    
    def add_agent_knowledge(self, agent_id: str, agent_data: Dict[str, Any]):
        """添加智能体知识"""
        self.graph.add_node(agent_id, 
                          type='agent',
                          attributes=agent_data,
                          updated_at=datetime.now().isoformat())
        
        # 连接相关经济概念
        self.graph.add_edge(agent_id, 'risk_preference', 
                          relationship='has_attribute',
                          weight=agent_data.get('risk_preference', 0.5))
    
    def add_market_knowledge(self, market_id: str, market_data: Dict[str, Any]):
        """添加市场知识"""
        self.graph.add_node(market_id,
                          type='market',
                          attributes=market_data,
                          updated_at=datetime.now().isoformat())
        
        # 建立市场关系
        for commodity in market_data.get('commodities', []):
            self.graph.add_edge(market_id, commodity,
                              relationship='trades',
                              volume=market_data.get('volume', 0))
    
    def search(self, entity_type: str = None, attributes: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """搜索知识图谱"""
        results = []
        
        for node_id, node_data in self.graph.nodes(data=True):
            if entity_type and node_data['type'] != entity_type:
                continue
                
            if attributes:
                match = all(
                    node_data['attributes'].get(k) == v 
                    for k, v in attributes.items()
                )
                if not match:
                    continue
            
            results.append({
                'id': node_id,
                'type': node_data['type'],
                'attributes': node_data['attributes']
            })
        
        return results
    
    def get_recommendations(self, agent_id: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基于知识图谱的个性化推荐"""
        recommendations = []
        
        # 基于智能体历史行为和市场状态推荐
        if self.graph.has_node(agent_id):
            agent_attrs = self.graph.nodes[agent_id]['attributes']
            
            # 查找相似智能体的策略
            similar_agents = self._find_similar_agents(agent_attrs)
            
            for similar_agent in similar_agents:
                # 获取成功策略
                successful_strategies = self._extract_strategies(similar_agent)
                recommendations.extend(successful_strategies)
        
        return recommendations
    
    def _find_similar_agents(self, agent_attrs: Dict[str, Any]) -> List[str]:
        """寻找相似智能体"""
        similar = []
        target_risk = agent_attrs.get('risk_preference', 0.5)
        
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data['type'] == 'agent' and node_id != agent_attrs.get('id'):
                risk = node_data['attributes'].get('risk_preference', 0.5)
                if abs(risk - target_risk) < 0.2:
                    similar.append(node_id)
        
        return similar
    
    def _extract_strategies(self, agent_id: str) -> List[Dict[str, Any]]:
        """提取成功策略"""
        # 从知识图谱中提取策略模式
        strategies = []
        
        # 简化的策略提取逻辑
        if self.graph.has_node(agent_id):
            agent_data = self.graph.nodes[agent_id]['attributes']
            if 'successful_actions' in agent_data:
                strategies.extend(agent_data['successful_actions'])
        
        return strategies
    
    def export_knowledge(self, format_type: str = 'json') -> str:
        """导出知识图谱"""
        if format_type == 'json':
            data = {
                'nodes': dict(self.graph.nodes(data=True)),
                'edges': list(self.graph.edges(data=True))
            }
            return json.dumps(data, indent=2, default=str)
        return ""