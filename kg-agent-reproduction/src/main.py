import asyncio
import argparse
import logging
from typing import Dict, List, Any
import json

from kg_agent.core.agent import KGAgent
from kg_agent.core.memory import MemorySystem
from kg_agent.core.knowledge import KnowledgeGraphInterface

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def create_demo_agent():
    """创建演示智能体"""
    
    config = {
        'reasoning': {
            'max_search_depth': 3,
            'similarity_threshold': 0.8,
            'confidence_threshold': 0.7
        },
        'memory': {
            'decay_rate': 0.1,
            'reinforcement_threshold': 3,
            'max_memory_size': 1000
        },
        'knowledge': {
            'graph_path': 'data/knowledge_graph.json',
            'embedding_dim': 128
        }
    }
    
    agent = KGAgent("demo_agent", config)
    
    # 初始化知识图谱
    kg = agent.knowledge_interface
    
    # 添加示例数据
    await kg.add_entity("Albert_Einstein", {
        'type': 'person',
        'birth_year': 1879,
        'nationality': 'German',
        'profession': 'physicist'
    })
    
    await kg.add_entity("Theory_of_Relativity", {
        'type': 'concept',
        'field': 'physics',
        'year': 1905
    })
    
    await kg.add_relation("Albert_Einstein", "Theory_of_Relativity", 
                       "developed", {'confidence': 0.95})
    
    return agent

async def run_demo_tasks(agent: KGAgent):
    """运行演示任务"""
    
    tasks = [
        {
            'type': 'path_finding',
            'description': 'Find the connection between Albert Einstein and Theory of Relativity',
            'entities': ['Albert_Einstein', 'Theory_of_Relativity']
        },
        {
            'type': 'relation_prediction',
            'description': 'What is the relationship between Albert Einstein and Theory of Relativity?',
            'entities': ['Albert_Einstein', 'Theory_of_Relativity']
        },
        {
            'type': 'complex_reasoning',
            'description': 'Explain how Albert Einstein developed the Theory of Relativity',
            'entities': ['Albert_Einstein', 'Theory_of_Relativity']
        }
    ]
    
    results = []
    
    for i, task in enumerate(tasks):
        logger.info(f"Processing task {i+1}: {task['description']}")
        
        result = await agent.process_task(task)
        results.append({
            'task_id': i+1,
            'task': task,
            'result': result
        })
        
        logger.info(f"Task {i+1} completed: success={result['success']}, "
                   f"confidence={result.get('confidence', 0.0):.2f}")
    
    return results

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='KG-Agent Reproduction Demo')
    parser.add_argument('--mode', choices=['demo', 'interactive'], default='demo')
    parser.add_argument('--output', type=str, default='outputs')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        logger.info("Starting KG-Agent demo...")
        
        agent = await create_demo_agent()
        results = await run_demo_tasks(agent)
        
        # 保存结果
        import os
        os.makedirs(args.output, exist_ok=True)
        
        with open(f"{args.output}/demo_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # 打印统计
        stats = agent.get_memory_stats() if hasattr(agent, 'get_memory_stats') else {}
        logger.info(f"Memory stats: {stats}")
        
        logger.info("Demo completed! Results saved to outputs/demo_results.json")
    
    elif args.mode == 'interactive':
        logger.info("Interactive mode - please implement custom logic")
        # 这里可以添加交互式模式

if __name__ == "__main__":
    asyncio.run(main())