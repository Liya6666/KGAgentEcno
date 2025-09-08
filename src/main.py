#!/usr/bin/env python3
"""
MAGE: Multi-Agent Knowledge Graph Economics
基于多智能体和知识图谱的经济仿真系统主程序
"""

import argparse
import os
import sys
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.agent.economic_agent import EconomicAgent, AgentType, AgentState
from core.knowledge.economic_kg import EconomicKnowledgeGraph
from core.economics.market import Economy
from core.simulation.engine import SimulationEngine, SimulationConfig

def create_demo_economy():
    """创建演示经济系统"""
    
    # 创建知识图谱
    kg = EconomicKnowledgeGraph()
    
    # 创建经济系统
    economy = Economy("Demo_Economy")
    
    # 添加市场
    economy.add_market("food", 50.0)
    economy.add_market("labor", 100.0)
    economy.add_market("capital", 1000.0)
    
    # 创建智能体
    agents = []
    
    # 消费者智能体
    for i in range(3):
        state = AgentState(
            wealth=1000.0,
            goods={'food': 10.0},
            preferences={'food': 0.8, 'leisure': 0.2},
            knowledge_level=0.7,
            risk_preference=0.3,
            trust_score=0.8
        )
        agent = EconomicAgent(f"consumer_{i}", AgentType.CONSUMER, state, kg)
        economy.add_agent(agent.id, agent)
        agents.append(agent)
    
    # 生产者智能体
    for i in range(2):
        state = AgentState(
            wealth=5000.0,
            goods={'food': 100.0, 'labor': 50.0},
            preferences={'profit': 0.9, 'growth': 0.1},
            knowledge_level=0.8,
            risk_preference=0.6,
            trust_score=0.7
        )
        agent = EconomicAgent(f"producer_{i}", AgentType.PRODUCER, state, kg)
        economy.add_agent(agent.id, agent)
        agents.append(agent)
    
    return economy

def run_simulation():
    """运行仿真"""
    
    # 创建输出目录
    output_dir = f"outputs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建经济系统
    economy = create_demo_economy()
    
    # 配置仿真
    config = SimulationConfig(
        name="Economic_Simulation_Demo",
        num_steps=100,
        step_interval=0.1,
        save_interval=10,
        output_dir=output_dir
    )
    
    # 运行仿真
    engine = SimulationEngine(config)
    engine.initialize(economy)
    engine.run()
    
    # 输出摘要
    summary = engine.get_summary()
    print("\n=== 仿真结果摘要 ===")
    print(f"总财富: ${summary['total_wealth']:.2f}")
    print(f"总交易数: {summary['total_trades']}")
    print(f"最终价格: {summary['final_prices']}")
    print(f"财富不平等指数: {summary['wealth_inequality']:.2f}")
    
    return engine

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MAGE Economic Simulation")
    parser.add_argument("--mode", choices=["demo", "custom"], default="demo",
                       help="运行模式")
    parser.add_argument("--steps", type=int, default=100,
                       help="仿真步数")
    parser.add_argument("--output", type=str, default="outputs",
                       help="输出目录")
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        print("运行演示模式...")
        engine = run_simulation()
    else:
        print("自定义模式待实现...")
    
    print("\n仿真完成！结果已保存到 outputs/ 目录")

if __name__ == "__main__":
    main()