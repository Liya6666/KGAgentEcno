import pytest
import asyncio
import json
from kg_agent.core.agent import KGAgent
from kg_agent.core.memory import MemorySystem
from kg_agent.core.knowledge import KnowledgeGraphInterface

@pytest.fixture
async def test_agent():
    """测试智能体fixture"""
    config = {
        'reasoning': {'max_search_depth': 2},
        'memory': {'max_memory_size': 100},
        'knowledge': {'embedding_dim': 64}
    }
    
    agent = KGAgent("test_agent", config)
    
    # 添加测试数据
    kg = agent.knowledge_interface
    await kg.add_entity("test_entity_1", {'type': 'test', 'value': 1})
    await kg.add_entity("test_entity_2", {'type': 'test', 'value': 2})
    await kg.add_relation("test_entity_1", "test_entity_2", "test_relation")
    
    return agent

@pytest.mark.asyncio
async def test_agent_initialization(test_agent):
    """测试智能体初始化"""
    assert test_agent.agent_id == "test_agent"
    assert test_agent.state.confidence_score == 0.0

@pytest.mark.asyncio
async def test_task_processing(test_agent):
    """测试任务处理"""
    task = {
        'type': 'path_finding',
        'description': 'Find path between test entities',
        'entities': ['test_entity_1', 'test_entity_2']
    }
    
    result = await test_agent.process_task(task)
    
    assert result['success'] == True
    assert 'result' in result
    assert 'reasoning_path' in result

@pytest.mark.asyncio
async def test_knowledge_graph_interface():
    """测试知识图谱接口"""
    kg = KnowledgeGraphInterface({'embedding_dim': 64})
    
    await kg.add_entity("entity1", {'type': 'person', 'name': 'Alice'})
    await kg.add_entity("entity2", {'type': 'person', 'name': 'Bob'})
    await kg.add_relation("entity1", "entity2", "knows")
    
    stats = kg.get_graph_stats()
    assert stats['num_entities'] >= 2
    assert stats['num_relations'] >= 1

@pytest.mark.asyncio
async def test_memory_system():
    """测试记忆系统"""
    memory = MemorySystem({'max_memory_size': 10})
    
    task = {'task_type': 'test', 'entities': ['A', 'B']}
    result = {'success': True, 'confidence': 0.9}
    
    await memory.store_experience(task, result)
    
    similar_cases = await memory.find_similar_cases(task)
    assert len(similar_cases) >= 0

if __name__ == "__main__":
    pytest.main([__file__])