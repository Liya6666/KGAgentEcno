import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class MarketInfo:
    """市场信息"""
    price: float
    volume: float
    volatility: float
    trend: str  # 'up', 'down', 'stable'
    liquidity: float

class MarketMechanism(ABC):
    """市场机制抽象基类"""
    
    @abstractmethod
    def calculate_price(self, supply: float, demand: float) -> float:
        pass
    
    @abstractmethod
    def execute_trade(self, buyer_order: Dict[str, Any], 
                     seller_order: Dict[str, Any]) -> Dict[str, Any]:
        pass

class SupplyDemandMarket(MarketMechanism):
    """供需市场模型"""
    
    def __init__(self, commodity: str, initial_price: float = 100.0):
        self.commodity = commodity
        self.price = initial_price
        self.order_book = {'buy': [], 'sell': []}
        self.trade_history = []
        
    def calculate_price(self, supply: float, demand: float) -> float:
        """基于供需关系计算价格"""
        if supply == 0:
            return self.price * 1.5  # 稀缺溢价
        
        ratio = demand / supply if supply > 0 else 1.0
        
        # 价格弹性模型
        elasticity = 0.1  # 价格弹性系数
        price_change = elasticity * np.log(ratio)
        
        new_price = self.price * (1 + price_change)
        return max(new_price, 0.01)  # 防止负价格
    
    def submit_order(self, agent_id: str, order_type: str, 
                    quantity: float, price: Optional[float] = None):
        """提交订单"""
        order = {
            'agent_id': agent_id,
            'type': order_type,
            'quantity': quantity,
            'price': price or self.price,
            'timestamp': len(self.trade_history)
        }
        
        self.order_book[order_type].append(order)
        
        # 尝试匹配订单
        return self._match_orders()
    
    def _match_orders(self) -> List[Dict[str, Any]]:
        """订单匹配算法"""
        trades = []
        
        # 简单价格优先匹配
        buy_orders = sorted(self.order_book['buy'], 
                          key=lambda x: x['price'], reverse=True)
        sell_orders = sorted(self.order_book['sell'], 
                           key=lambda x: x['price'])
        
        while buy_orders and sell_orders:
            buy = buy_orders[0]
            sell = sell_orders[0]
            
            if buy['price'] >= sell['price']:
                # 成交
                trade_quantity = min(buy['quantity'], sell['quantity'])
                trade_price = (buy['price'] + sell['price']) / 2
                
                trade = {
                    'buyer': buy['agent_id'],
                    'seller': sell['agent_id'],
                    'quantity': trade_quantity,
                    'price': trade_price,
                    'timestamp': len(self.trade_history)
                }
                
                trades.append(trade)
                self.trade_history.append(trade)
                
                # 更新订单
                buy['quantity'] -= trade_quantity
                sell['quantity'] -= trade_quantity
                
                if buy['quantity'] <= 0:
                    buy_orders.pop(0)
                if sell['quantity'] <= 0:
                    sell_orders.pop(0)
            else:
                break
        
        # 更新订单簿
        self.order_book['buy'] = [o for o in buy_orders if o['quantity'] > 0]
        self.order_book['sell'] = [o for o in sell_orders if o['quantity'] > 0]
        
        # 更新市场价格
        if trades:
            self.price = np.mean([t['price'] for t in trades[-5:]])
        
        return trades
    
    def get_market_info(self) -> MarketInfo:
        """获取市场信息"""
        if len(self.trade_history) >= 5:
            recent_prices = [t['price'] for t in self.trade_history[-5:]]
            volatility = np.std(recent_prices)
            trend = 'up' if recent_prices[-1] > recent_prices[0] else 'down' if recent_prices[-1] < recent_prices[0] else 'stable'
            volume = sum([t['quantity'] for t in self.trade_history[-5:]])
        else:
            volatility = 0.1
            trend = 'stable'
            volume = 0
        
        return MarketInfo(
            price=self.price,
            volume=volume,
            volatility=volatility,
            trend=trend,
            liquidity=len(self.order_book['buy']) + len(self.order_book['sell'])
        )

class Economy:
    """经济系统"""
    
    def __init__(self, name: str):
        self.name = name
        self.markets = {}
        self.agents = {}
        self.gdp = 0.0
        self.inflation = 0.0
        self.unemployment = 0.0
        
    def add_market(self, commodity: str, initial_price: float = 100.0):
        """添加市场"""
        self.markets[commodity] = SupplyDemandMarket(commodity, initial_price)
        
    def add_agent(self, agent_id: str, agent):
        """添加智能体"""
        self.agents[agent_id] = agent
        
    def simulate_step(self):
        """仿真一步"""
        # 收集所有智能体的决策
        decisions = {}
        for agent_id, agent in self.agents.items():
            for commodity, market in self.markets.items():
                market_info = market.get_market_info()
                perception = agent.perceive({
                    'commodity': commodity,
                    'market_info': market_info.__dict__
                })
                decision = agent.decide(perception)
                decisions[f"{agent_id}_{commodity}"] = (agent, market, decision)
        
        # 执行决策
        results = {}
        for key, (agent, market, decision) in decisions.items():
            result = agent.act(decision, market)
            results[key] = result
            
            # 智能体学习
            agent.learn()
        
        # 更新经济指标
        self._update_economic_indicators()
        
        return results
    
    def _update_economic_indicators(self):
        """更新经济指标"""
        total_wealth = sum([agent.state.wealth for agent in self.agents.values()])
        self.gdp = total_wealth
        
        # 计算通胀（基于价格变化）
        if hasattr(self, '_previous_prices'):
            current_prices = {k: m.price for k, m in self.markets.items()}
            price_changes = [
                (current_prices[k] - self._previous_prices[k]) / self._previous_prices[k]
                for k in current_prices
            ]
            self.inflation = np.mean(price_changes) if price_changes else 0.0
            self._previous_prices = current_prices
        else:
            self._previous_prices = {k: m.price for k, m in self.markets.items()}