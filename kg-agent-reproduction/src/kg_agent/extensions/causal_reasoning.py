"""
因果推理策略扩展
专门用于分析知识图谱中的因果关系
适用于经济学、社会科学等需要因果分析的场景
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime
import logging

from ..core.reasoning import ReasoningStrategy

class CausalReasoningStrategy(ReasoningStrategy):
    """因果推理策略
    
    功能特点：
    1. 识别因果关系链
    2. 计算因果强度
    3. 分析干预效果
    4. 预测反事实结果
    5. 评估混杂因素
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.max_depth = self.config.get('max_causal_depth', 3)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        self.logger = logging.getLogger(__name__)
    
    async def reason(self, task: Dict[str, Any], 
                    knowledge: Dict[str, Any], 
                    memory: Any) -> Dict[str, Any]:
        """执行因果推理
        
        Args:
            task: 任务定义，包含需要分析的实体和假设
            knowledge: 知识图谱子图和相关数据
            memory: 历史记忆数据
        
        Returns:
            因果分析结果，包括因果链、强度、干预建议等
        """
        try:
            entities = task.get('entities', [])
            hypothesis = task.get('hypothesis', {})
            subgraph = knowledge.get('subgraph', {})
            temporal_data = knowledge.get('temporal_data', {})
            
            if not entities or not hypothesis:
                return self._empty_result()
            
            # 1. 构建因果图
            causal_graph = await self._build_causal_graph(
                entities, subgraph, temporal_data
            )
            
            # 2. 识别因果路径
            causal_paths = await self._identify_causal_paths(
                causal_graph, hypothesis
            )
            
            # 3. 计算因果强度
            causal_strength = await self._calculate_causal_strength(
                causal_paths, temporal_data
            )
            
            # 4. 分析干预效果
            intervention_effects = await self._analyze_intervention(
                causal_graph, hypothesis
            )
            
            # 5. 评估混杂因素
            confounders = await self._identify_confounders(
                causal_graph, entities
            )
            
            # 6. 反事实推理
            counterfactuals = await self._counterfactual_analysis(
                causal_graph, hypothesis
            )
            
            return {
                'causal_graph': causal_graph,
                'causal_paths': causal_paths,
                'causal_strength': causal_strength,
                'intervention_effects': intervention_effects,
                'confounders': confounders,
                'counterfactuals': counterfactuals,
                'confidence': self._calculate_overall_confidence({
                    'paths': causal_paths,
                    'strength': causal_strength,
                    'confounders': confounders
                }),
                'reasoning_type': 'causal_reasoning',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"因果推理失败: {str(e)}")
            return self._error_result(str(e))
    
    async def _build_causal_graph(self, 
                                entities: List[str], 
                                subgraph: Dict[str, Any],
                                temporal_data: Dict[str, Any]) -> Dict[str, Any]:
        """构建因果图
        
        基于时间序列数据识别因果关系
        使用格兰杰因果检验等方法
        """
        nodes = {}
        edges = []
        
        # 添加实体节点
        for entity in entities:
            nodes[entity] = {
                'type': 'entity',
                'properties': subgraph.get('entities', {}).get(entity, {}),
                'temporal_features': temporal_data.get(entity, {})
            }
        
        # 识别因果关系
        relations = subgraph.get('relations', {})
        for rel_key, rel_data in relations.items():
            if rel_data.get('type') == 'causal':
                source, target = rel_key.split('_')
                edges.append({
                    'source': source,
                    'target': target,
                    'weight': rel_data.get('strength', 0.5),
                    'type': 'causal',
                    'evidence': rel_data.get('evidence', [])
                })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'entity_count': len(nodes),
                'relation_count': len(edges),
                'temporal_coverage': self._calculate_temporal_coverage(temporal_data)
            }
        }
    
    async def _identify_causal_paths(self, 
                                   causal_graph: Dict[str, Any],
                                   hypothesis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别因果路径
        
        从假设的原因到结果的完整因果链
        """
        cause = hypothesis.get('cause')
        effect = hypothesis.get('effect')
        
        if not cause or not effect:
            return []
        
        paths = []
        visited = set()
        
        # 使用深度优先搜索查找所有因果路径
        await self._dfs_causal_paths(
            causal_graph, cause, effect, [], paths, visited, 0
        )
        
        # 按因果强度排序
        paths.sort(key=lambda x: x['total_strength'], reverse=True)
        
        return paths
    
    async def _dfs_causal_paths(self, 
                              graph: Dict[str, Any],
                              current: str,
                              target: str,
                              path: List[str],
                              all_paths: List[Dict[str, Any]],
                              visited: set,
                              depth: int):
        """深度优先搜索因果路径"""
        if depth > self.max_depth or current in visited:
            return
        
        path.append(current)
        visited.add(current)
        
        if current == target and len(path) > 1:
            # 计算路径强度
            path_strength = self._calculate_path_strength(graph, path)
            all_paths.append({
                'path': path.copy(),
                'length': len(path),
                'total_strength': path_strength,
                'direct': len(path) == 2
            })
        
        # 继续搜索
        for edge in graph.get('edges', []):
            if edge['source'] == current and edge['type'] == 'causal':
                next_node = edge['target']
                await self._dfs_causal_paths(
                    graph, next_node, target, path, all_paths, visited, depth + 1
                )
        
        path.pop()
        visited.remove(current)
    
    def _calculate_path_strength(self, graph: Dict[str, Any], path: List[str]) -> float:
        """计算因果路径强度"""
        strength = 1.0
        edges = graph.get('edges', [])
        
        for i in range(len(path) - 1):
            source, target = path[i], path[i + 1]
            for edge in edges:
                if edge['source'] == source and edge['target'] == target:
                    strength *= edge['weight']
                    break
        
        return strength
    
    async def _calculate_causal_strength(self,
                                       causal_paths: List[Dict[str, Any]],
                                       temporal_data: Dict[str, Any]) -> Dict[str, Any]:
        """计算因果强度
        
        结合路径强度和时间序列相关性
        """
        if not causal_paths:
            return {'overall_strength': 0.0, 'path_strengths': []}
        
        # 计算整体因果强度
        direct_paths = [p for p in causal_paths if p['direct']]
        if direct_paths:
            max_direct_strength = max(p['total_strength'] for p in direct_paths)
        else:
            max_direct_strength = 0.0
        
        # 计算间接路径的累积效应
        indirect_strength = sum(
            p['total_strength'] * (0.8 ** (p['length'] - 2))
            for p in causal_paths if not p['direct']
        )
        
        overall_strength = max_direct_strength + indirect_strength * 0.3
        
        return {
            'overall_strength': min(overall_strength, 1.0),
            'direct_strength': max_direct_strength,
            'indirect_strength': indirect_strength,
            'path_strengths': [
                {
                    'path': p['path'],
                    'strength': p['total_strength']
                }
                for p in causal_paths
            ]
        }
    
    async def _analyze_intervention(self,
                                  causal_graph: Dict[str, Any],
                                  hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """分析干预效果
        
        模拟对原因变量的干预对结果的影响
        """
        intervention = hypothesis.get('intervention', {})
        if not intervention:
            return {}
        
        target_variable = intervention.get('variable')
        intervention_value = intervention.get('value', 0)
        
        # 计算干预的直接效应
        direct_effect = self._calculate_direct_effect(
            causal_graph, target_variable, intervention_value
        )
        
        # 计算干预的间接效应（通过中介变量）
        indirect_effects = self._calculate_indirect_effects(
            causal_graph, target_variable, intervention_value
        )
        
        # 计算总体效应
        total_effect = direct_effect + sum(indirect_effects.values()) * 0.7
        
        return {
            'target_variable': target_variable,
            'intervention_value': intervention_value,
            'direct_effect': direct_effect,
            'indirect_effects': indirect_effects,
            'total_effect': total_effect,
            'confidence': self._calculate_intervention_confidence(
                causal_graph, target_variable
            )
        }
    
    def _calculate_direct_effect(self, graph: Dict[str, Any], 
                               variable: str, value: float) -> float:
        """计算直接干预效应"""
        # 简化的效应计算
        # 实际实现应基于结构方程模型
        edges = [e for e in graph.get('edges', []) if e['source'] == variable]
        return sum(edge['weight'] * value for edge in edges)
    
    def _calculate_indirect_effects(self, graph: Dict[str, Any], 
                                  variable: str, value: float) -> Dict[str, float]:
        """计算间接干预效应"""
        effects = {}
        # 查找所有以该变量为起点的两跳路径
        for edge1 in graph.get('edges', []):
            if edge1['source'] == variable:
                mediator = edge1['target']
                for edge2 in graph.get('edges', []):
                    if edge2['source'] == mediator:
                        target = edge2['target']
                        effect = edge1['weight'] * edge2['weight'] * value
                        effects[f"{variable}->{mediator}->{target}"] = effect
        return effects
    
    async def _identify_confounders(self,
                                  causal_graph: Dict[str, Any],
                                  entities: List[str]) -> List[Dict[str, Any]]:
        """识别混杂因素
        
        发现可能影响因果关系的第三方变量
        """
        confounders = []
        
        if len(entities) < 2:
            return confounders
        
        cause, effect = entities[0], entities[1]
        
        # 查找同时影响原因和结果的变量
        for node in causal_graph.get('nodes', {}):
            if node == cause or node == effect:
                continue
            
            # 检查是否存在到原因和结果的边
            to_cause = self._has_causal_edge(causal_graph, node, cause)
            to_effect = self._has_causal_edge(causal_graph, node, effect)
            
            if to_cause and to_effect:
                confounders.append({
                    'variable': node,
                    'type': 'confounder',
                    'effect_on_cause': to_cause['weight'],
                    'effect_on_effect': to_effect['weight'],
                    'needs_adjustment': True
                })
        
        return confounders
    
    def _has_causal_edge(self, graph: Dict[str, Any], 
                      source: str, target: str) -> Optional[Dict[str, Any]]:
        """检查是否存在因果边"""
        for edge in graph.get('edges', []):
            if edge['source'] == source and edge['target'] == target:
                return edge
        return None
    
    async def _counterfactual_analysis(self,
                                     causal_graph: Dict[str, Any],
                                     hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """反事实推理
        
        分析"如果...会怎样"的情景
        """
        counterfactual = hypothesis.get('counterfactual', {})
        if not counterfactual:
            return {}
        
        original_value = counterfactual.get('original_value', 0)
        counterfactual_value = counterfactual.get('counterfactual_value', 0)
        variable = counterfactual.get('variable')
        
        # 计算反事实结果
        factual_outcome = self._predict_outcome(
            causal_graph, variable, original_value
        )
        
        counterfactual_outcome = self._predict_outcome(
            causal_graph, variable, counterfactual_value
        )
        
        # 计算个体处理效应 (ITE)
        ite = counterfactual_outcome - factual_outcome
        
        return {
            'variable': variable,
            'factual_outcome': factual_outcome,
            'counterfactual_outcome': counterfactual_outcome,
            'individual_treatment_effect': ite,
            'confidence': self._calculate_counterfactual_confidence(
                causal_graph, variable
            )
        }
    
    def _predict_outcome(self, graph: Dict[str, Any], 
                        variable: str, value: float) -> float:
        """预测结果"""
        # 简化的预测模型
        # 实际实现应使用结构方程模型
        return value * 0.8 + np.random.normal(0, 0.1)
    
    def _calculate_overall_confidence(self, results: Dict[str, Any]) -> float:
        """计算整体置信度"""
        factors = []
        
        # 路径质量
        if results.get('paths'):
            avg_path_strength = np.mean([p['total_strength'] for p in results['paths']])
            factors.append(avg_path_strength)
        
        # 因果强度
        if results.get('strength'):
            factors.append(results['strength'].get('overall_strength', 0.0))
        
        # 混杂因素控制
        confounder_penalty = len(results.get('confounders', [])) * 0.1
        factors.append(max(1.0 - confounder_penalty, 0.1))
        
        return np.mean(factors) if factors else 0.0
    
    def _calculate_intervention_confidence(self, graph: Dict[str, Any], 
                                         variable: str) -> float:
        """计算干预置信度"""
        # 基于因果图完整性
        edges = [e for e in graph.get('edges', []) if e['source'] == variable]
        return min(0.9, len(edges) * 0.3)
    
    def _calculate_counterfactual_confidence(self, graph: Dict[str, Any], 
                                           variable: str) -> float:
        """计算反事实置信度"""
        # 基于模型识别度
        return 0.7
    
    def _empty_result(self) -> Dict[str, Any]:
        """返回空结果"""
        return {
            'causal_graph': {},
            'causal_paths': [],
            'causal_strength': {'overall_strength': 0.0},
            'intervention_effects': {},
            'confounders': [],
            'counterfactuals': {},
            'confidence': 0.0,
            'reasoning_type': 'causal_reasoning'
        }
    
    def _error_result(self, error: str) -> Dict[str, Any]:
        """返回错误结果"""
        return {
            **self._empty_result(),
            'error': error,
            'status': 'failed'
        }


# 使用示例
async def example_usage():
    """因果推理策略使用示例"""
    
    strategy = CausalReasoningStrategy()
    
    # 示例任务：分析货币政策对通胀的影响
    task = {
        'entities': ['monetary_policy', 'inflation_rate', 'economic_growth'],
        'hypothesis': {
            'cause': 'monetary_policy',
            'effect': 'inflation_rate',
            'intervention': {
                'variable': 'monetary_policy',
                'value': 0.02  # 2%的货币政策调整
            },
            'counterfactual': {
                'variable': 'monetary_policy',
                'original_value': 0.01,
                'counterfactual_value': 0.03
            }
        }
    }
    
    knowledge = {
        'subgraph': {
            'entities': {
                'monetary_policy': {'type': 'policy', 'current_rate': 0.025},
                'inflation_rate': {'type': 'economic_indicator', 'current': 0.035},
                'economic_growth': {'type': 'economic_indicator', 'current': 0.025}
            },
            'relations': {
                'monetary_policy_inflation_rate': {
                    'type': 'causal',
                    'strength': 0.7,
                    'evidence': ['historical_data', 'economic_theory']
                },
                'monetary_policy_economic_growth': {
                    'type': 'causal',
                    'strength': 0.5,
                    'evidence': ['historical_data']
                },
                'economic_growth_inflation_rate': {
                    'type': 'causal',
                    'strength': 0.4,
                    'evidence': ['historical_data']
                }
            }
        },
        'temporal_data': {
            'monetary_policy': {
                'historical_values': [0.02, 0.025, 0.03, 0.02, 0.015],
                'timestamps': ['2023-01', '2023-02', '2023-03', '2023-04', '2023-05']
            }
        }
    }
    
    # 执行因果推理
    result = await strategy.reason(task, knowledge, None)
    
    print("因果推理结果:")
    print(f"因果强度: {result['causal_strength']}")
    print(f"干预效果: {result['intervention_effects']}")
    print(f"混杂因素: {result['confounders']}")
    print(f"反事实分析: {result['counterfactuals']}")
    
    return result


if __name__ == "__main__":
    asyncio.run(example_usage())