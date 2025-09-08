import time
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
from dataclasses import dataclass

@dataclass
class SimulationConfig:
    """仿真配置"""
    name: str
    num_steps: int
    step_interval: float  # 秒
    save_interval: int
    output_dir: str
    random_seed: Optional[int] = None

class SimulationEngine:
    """经济仿真引擎"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.economy = None
        self.results = []
        self.current_step = 0
        self.start_time = None
        
    def initialize(self, economy):
        """初始化仿真"""
        self.economy = economy
        self.start_time = datetime.now()
        
        # 初始化结果存储
        self.results = []
        
    def run(self):
        """运行仿真"""
        print(f"Starting simulation: {self.config.name}")
        print(f"Steps: {self.config.num_steps}")
        
        for step in range(self.config.num_steps):
            self.current_step = step
            
            # 执行一步仿真
            step_result = self.step()
            
            # 保存结果
            self.results.append({
                'step': step,
                'timestamp': datetime.now().isoformat(),
                'data': step_result
            })
            
            # 定期保存
            if step % self.config.save_interval == 0:
                self.save_checkpoint(step)
            
            # 控制仿真速度
            if self.config.step_interval > 0:
                time.sleep(self.config.step_interval)
        
        # 保存最终结果
        self.save_final_results()
        
        print("Simulation completed!")
        
    def step(self) -> Dict[str, Any]:
        """执行一步仿真"""
        # 执行经济仿真
        market_results = self.economy.simulate_step()
        
        # 收集统计数据
        stats = self._collect_statistics()
        
        return {
            'market_results': market_results,
            'statistics': stats,
            'step': self.current_step
        }
    
    def _collect_statistics(self) -> Dict[str, Any]:
        """收集统计数据"""
        stats = {
            'gdp': self.economy.gdp,
            'inflation': self.economy.inflation,
            'unemployment': self.economy.unemployment,
            'num_agents': len(self.economy.agents),
            'markets': {}
        }
        
        # 市场统计
        for commodity, market in self.economy.markets.items():
            market_info = market.get_market_info()
            stats['markets'][commodity] = {
                'price': market_info.price,
                'volume': market_info.volume,
                'volatility': market_info.volatility,
                'trend': market_info.trend
            }
        
        # 智能体统计
        agent_wealth = [agent.state.wealth for agent in self.economy.agents.values()]
        stats['wealth_distribution'] = {
            'mean': np.mean(agent_wealth) if agent_wealth else 0,
            'std': np.std(agent_wealth) if agent_wealth else 0,
            'min': min(agent_wealth) if agent_wealth else 0,
            'max': max(agent_wealth) if agent_wealth else 0
        }
        
        return stats
    
    def save_checkpoint(self, step: int):
        """保存检查点"""
        checkpoint = {
            'step': step,
            'economy_state': {
                'gdp': self.economy.gdp,
                'inflation': self.economy.inflation,
                'markets': {k: v.price for k, v in self.economy.markets.items()}
            }
        }
        
        filename = f"{self.config.output_dir}/checkpoint_{step}.json"
        with open(filename, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def save_final_results(self):
        """保存最终结果"""
        final_results = {
            'config': self.config.__dict__,
            'total_steps': len(self.results),
            'final_statistics': self._collect_statistics(),
            'results': self.results
        }
        
        filename = f"{self.config.output_dir}/simulation_results.json"
        with open(filename, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # 保存为CSV格式
        df = pd.DataFrame([
            {
                'step': r['step'],
                'gdp': r['data']['statistics']['gdp'],
                'inflation': r['data']['statistics']['inflation'],
                **{f"{k}_price": v['price'] 
                   for k, v in r['data']['statistics']['markets'].items()}
            }
            for r in self.results
        ])
        
        csv_filename = f"{self.config.output_dir}/simulation_data.csv"
        df.to_csv(csv_filename, index=False)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取仿真摘要"""
        if not self.results:
            return {}
        
        final_stats = self.results[-1]['data']['statistics']
        
        return {
            'total_wealth': final_stats['gdp'],
            'total_trades': len(self.results),
            'final_prices': {
                k: v['price'] 
                for k, v in final_stats['markets'].items()
            },
            'wealth_inequality': final_stats['wealth_distribution']['std']
        }