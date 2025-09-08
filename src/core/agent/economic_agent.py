import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class AgentType(Enum):
    CONSUMER = "consumer"
    PRODUCER = "producer"
    INVESTOR = "investor"
    GOVERNMENT = "government"
    BANK = "bank"

@dataclass
class AgentState:
    """智能体状态"""
    wealth: float
    goods: Dict[str, float]
    preferences: Dict[str, float]
    knowledge_level: float
    risk_preference: float
    trust_score: float

class EconomicAgent:
    """
    经济智能体 - 基于强化学习的决策智能体
    结合知识图谱推理和经济学理论
    """
    
    def __init__(self, agent_id: str, agent_type: AgentType, 
                 initial_state: AgentState, knowledge_graph=None):
        self.id = agent_id
        self.type = agent_type
        self.state = initial_state
        self.knowledge_graph = knowledge_graph
        self.memory = []
        self.strategy = {}
        self.rl_model = None
        
    def perceive(self, market_state: Dict[str, Any]) -> Dict[str, Any]:
        """感知市场环境"""
        # 从知识图谱获取相关信息
        relevant_knowledge = self._query_knowledge_graph(market_state)
        
        # 整合感知信息
        perception = {
            'market_state': market_state,
            'knowledge': relevant_knowledge,
            'personal_state': self.state.__dict__
        }
        
        return perception
    
    def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """基于强化学习的决策"""
        # 使用Q-learning或策略梯度方法
        state_features = self._extract_features(perception)
        
        if self.rl_model:
            action = self.rl_model.predict(state_features)
        else:
            action = self._rule_based_action(state_features)
            
        return {
            'action_type': action['type'],
            'action_params': action['params'],
            'confidence': action.get('confidence', 0.5)
        }
    
    def act(self, decision: Dict[str, Any], market) -> Dict[str, Any]:
        """执行决策并更新状态"""
        # 执行市场操作
        result = market.execute_order(self.id, decision)
        
        # 更新智能体状态
        self._update_state(result)
        
        # 记录经验
        self.memory.append({
            'state': self.state.__dict__,
            'action': decision,
            'result': result
        })
        
        return result
    
    def learn(self):
        """从经验中学习"""
        if len(self.memory) > 10:
            experiences = self.memory[-10:]
            self._update_rl_model(experiences)
    
    def _query_knowledge_graph(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """查询知识图谱获取决策支持信息"""
        if not self.knowledge_graph:
            return {}
        
        # 基于查询获取相关经济知识
        results = self.knowledge_graph.search(
            entity_type="economic_entity",
            attributes=query
        )
        return results
    
    def _extract_features(self, perception: Dict[str, Any]) -> np.ndarray:
        """提取状态特征用于RL"""
        features = []
        
        # 市场特征
        market = perception['market_state']
        features.extend([
            market.get('price', 0),
            market.get('volatility', 0),
            market.get('volume', 0)
        ])
        
        # 个人特征
        personal = perception['personal_state']
        features.extend([
            personal['wealth'],
            personal['risk_preference'],
            personal['knowledge_level']
        ])
        
        return np.array(features)
    
    def _rule_based_action(self, features: np.ndarray) -> Dict[str, Any]:
        """基于规则的行动（用于冷启动）"""
        # 简单的基于规则的策略
        wealth, risk_pref = features[3], features[4]
        
        if self.type == AgentType.CONSUMER:
            return {'type': 'buy', 'params': {'quantity': min(wealth * 0.1, 100)}}
        elif self.type == AgentType.PRODUCER:
            return {'type': 'produce', 'params': {'amount': 50}}
        else:
            return {'type': 'hold', 'params': {}}