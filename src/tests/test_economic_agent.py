import pytest
import numpy as np
from src.core.agent.economic_agent import EconomicAgent, AgentType, AgentState
from src.core.knowledge.economic_kg import EconomicKnowledgeGraph

class TestEconomicAgent:
    
    def test_agent_initialization(self):
        """测试智能体初始化"""
        state = AgentState(
            wealth=1000.0,
            goods={'food': 10.0},
            preferences={'food': 0.8},
            knowledge_level=0.7,
            risk_preference=0.3,
            trust_score=0.8
        )
        
        agent = EconomicAgent("test_1", AgentType.CONSUMER, state)
        
        assert agent.id == "test_1"
        assert agent.type == AgentType.CONSUMER
        assert agent.state.wealth == 1000.0
    
    def test_perception(self):
        """测试感知功能"""
        state = AgentState(
            wealth=1000.0,
            goods={'food': 10.0},
            preferences={'food': 0.8},
            knowledge_level=0.7,
            risk_preference=0.3,
            trust_score=0.8
        )
        
        agent = EconomicAgent("test_1", AgentType.CONSUMER, state)
        
        market_state = {
            'price': 50.0,
            'volatility': 0.1,
            'volume': 100.0
        }
        
        perception = agent.perceive(market_state)
        
        assert 'market_state' in perception
        assert 'personal_state' in perception
        assert perception['market_state']['price'] == 50.0
    
    def test_decision_making(self):
        """测试决策制定"""
        state = AgentState(
            wealth=1000.0,
            goods={'food': 10.0},
            preferences={'food': 0.8},
            knowledge_level=0.7,
            risk_preference=0.3,
            trust_score=0.8
        )
        
        agent = EconomicAgent("test_1", AgentType.CONSUMER, state)
        
        perception = {
            'market_state': {'price': 50.0, 'volatility': 0.1},
            'personal_state': state.__dict__
        }
        
        decision = agent.decide(perception)
        
        assert 'action_type' in decision
        assert 'action_params' in decision

if __name__ == "__main__":
    pytest.main([__file__])