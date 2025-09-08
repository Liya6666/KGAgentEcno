"""
因果推理策略使用示例
展示如何使用新的因果推理策略扩展
"""

import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.kg_agent.extensions.causal_reasoning import CausalReasoningStrategy
from typing import Dict, List, Any

class TaskType:
    """任务类型枚举"""
    PATH_FINDING = "path_finding"
    RELATION_PREDICTION = "relation_prediction"
    COMPLEX_REASONING = "complex_reasoning"
    CAUSAL_REASONING = "causal_reasoning"
    ENTITY_LINKING = "entity_linking"
    QUESTION_ANSWERING = "question_answering"


class ExtendedReasoningEngine:
    """扩展推理引擎，集成因果推理策略"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.strategies = {
            TaskType.PATH_FINDING: PathFindingStrategy(),
            TaskType.RELATION_PREDICTION: RelationPredictionStrategy(),
            TaskType.COMPLEX_REASONING: ComplexReasoningStrategy(),
            TaskType.CAUSAL_REASONING: CausalReasoningStrategy(self.config.get('causal', {})),
            TaskType.ENTITY_LINKING: ComplexReasoningStrategy(),
            TaskType.QUESTION_ANSWERING: ComplexReasoningStrategy()
        }
    
    async def reason(self, task: Dict[str, Any], 
                    knowledge: Dict[str, Any], 
                    memory: Any) -> Dict[str, Any]:
        """执行推理"""
        task_type = task.get('task_type', TaskType.COMPLEX_REASONING)
        
        if task_type not in self.strategies:
            task_type = TaskType.COMPLEX_REASONING
        
        strategy = self.strategies[task_type]
        return await strategy.reason(task, knowledge, memory)


async def economic_causal_analysis_example():
    """经济学因果分析示例"""
    
    # 初始化因果推理策略
    causal_strategy = CausalReasoningStrategy({
        'max_causal_depth': 4,
        'confidence_threshold': 0.7
    })
    
    # 示例：分析货币政策对通胀和经济增长的影响
    task = {
        'task_type': 'causal_reasoning',
        'entities': ['interest_rate', 'inflation', 'gdp_growth', 'employment'],
        'hypothesis': {
            'cause': 'interest_rate',
            'effect': 'inflation',
            'intervention': {
                'variable': 'interest_rate',
                'value': 0.025  # 加息250个基点
            },
            'counterfactual': {
                'variable': 'interest_rate',
                'original_value': 0.015,
                'counterfactual_value': 0.035
            }
        }
    }
    
    # 构建知识图谱数据
    knowledge = {
        'subgraph': {
            'entities': {
                'interest_rate': {
                    'type': 'monetary_policy',
                    'current_value': 0.025,
                    'historical_range': [0.001, 0.06]
                },
                'inflation': {
                    'type': 'price_level',
                    'current_value': 0.032,
                    'target': 0.02
                },
                'gdp_growth': {
                    'type': 'economic_indicator',
                    'current_value': 0.025
                },
                'employment': {
                    'type': 'labor_market',
                    'current_value': 0.038  # 失业率
                }
            },
            'relations': {
                'interest_rate_inflation': {
                    'type': 'causal',
                    'strength': -0.7,  # 负相关
                    'evidence': ['taylor_rule', 'empirical_data']
                },
                'interest_rate_gdp_growth': {
                    'type': 'causal',
                    'strength': -0.5,
                    'evidence': ['is_curve', 'empirical_data']
                },
                'gdp_growth_employment': {
                    'type': 'causal',
                    'strength': -0.6,  # 增长降低失业率
                    'evidence': ['okuns_law', 'empirical_data']
                },
                'gdp_growth_inflation': {
                    'type': 'causal',
                    'strength': 0.4,  # 菲利普斯曲线
                    'evidence': ['phillips_curve', 'empirical_data']
                }
            }
        },
        'temporal_data': {
            'interest_rate': {
                'quarterly_data': [0.02, 0.0225, 0.025, 0.0275],
                'annual_data': [0.018, 0.025, 0.03]
            },
            'inflation': {
                'quarterly_data': [0.035, 0.033, 0.032, 0.03],
                'annual_data': [0.036, 0.032, 0.028]
            }
        },
        'historical_cases': [
            {
                'period': '2008-2009',
                'interest_rate_change': -0.05,
                'inflation_change': -0.02,
                'gdp_growth_change': -0.03
            },
            {
                'period': '2020-2021',
                'interest_rate_change': -0.015,
                'inflation_change': 0.025,
                'gdp_growth_change': -0.02
            }
        ]
    }
    
    # 执行因果推理
    print("🔄 执行因果推理分析...")
    result = await causal_strategy.reason(task, knowledge, None)
    
    # 展示结果
    print("\n📊 因果推理结果:")
    print("=" * 50)
    
    print(f"\n🎯 因果强度:")
    print(f"  总体强度: {result['causal_strength']['overall_strength']:.3f}")
    print(f"  直接影响: {result['causal_strength']['direct_strength']:.3f}")
    print(f"  间接影响: {result['causal_strength']['indirect_strength']:.3f}")
    
    print(f"\n📈 干预效果分析:")
    intervention = result['intervention_effects']
    if intervention:
        print(f"  干预变量: {intervention['target_variable']}")
        print(f"  干预幅度: {intervention['intervention_value']:.3f}")
        print(f"  总体效应: {intervention['total_effect']:.3f}")
        print(f"  直接效应: {intervention['direct_effect']:.3f}")
        if intervention['indirect_effects']:
            print(f"  间接效应: {intervention['indirect_effects']}")
    
    print(f"\n🎭 反事实分析:")
    counterfactual = result['counterfactuals']
    if counterfactual:
        print(f"  事实结果: {counterfactual['factual_outcome']:.3f}")
        print(f"  反事实结果: {counterfactual['counterfactual_outcome']:.3f}")
        print(f"  个体处理效应: {counterfactual['individual_treatment_effect']:.3f}")
    
    print(f"\n⚠️ 混杂因素:")
    for confounder in result['confounders']:
        print(f"  - {confounder['variable']}: "
              f"对原因影响={confounder['effect_on_cause']:.2f}, "
              f"对结果影响={confounder['effect_on_effect']:.2f}")
    
    print(f"\n✅ 置信度: {result['confidence']:.3f}")
    
    return result


async def social_science_example():
    """社会科学因果分析示例"""
    
    causal_strategy = CausalReasoningStrategy()
    
    # 示例：分析教育政策对社会流动性的影响
    task = {
        'task_type': 'causal_reasoning',
        'entities': ['education_policy', 'social_mobility', 'income_inequality', 'access_to_education'],
        'hypothesis': {
            'cause': 'education_policy',
            'effect': 'social_mobility',
            'intervention': {
                'variable': 'education_policy',
                'value': 1.0  # 实施新的教育政策
            }
        }
    }
    
    knowledge = {
        'subgraph': {
            'entities': {
                'education_policy': {'type': 'policy', 'coverage': 0.8},
                'social_mobility': {'type': 'social_indicator', 'current_index': 0.65},
                'income_inequality': {'type': 'economic_indicator', 'gini_coefficient': 0.35},
                'access_to_education': {'type': 'social_factor', 'access_rate': 0.85}
            },
            'relations': {
                'education_policy_access_to_education': {
                    'type': 'causal',
                    'strength': 0.8,
                    'evidence': ['policy_evaluation', 'empirical_studies']
                },
                'access_to_education_social_mobility': {
                    'type': 'causal',
                    'strength': 0.7,
                    'evidence': ['longitudinal_studies', 'econometric_analysis']
                },
                'income_inequality_social_mobility': {
                    'type': 'causal',
                    'strength': -0.6,
                    'evidence': ['cross_country_analysis', 'panel_data']
                }
            }
        }
    }
    
    result = await causal_strategy.reason(task, knowledge, None)
    
    print("\n📚 社会科学因果分析结果:")
    print("=" * 50)
    print(f"教育政策对社会流动性的因果强度: {result['causal_strength']['overall_strength']:.3f}")
    print(f"政策干预预期效果: {result['intervention_effects'].get('total_effect', 'N/A')}")
    
    return result


async def main():
    """主函数：运行因果推理示例"""
    print("🔬 KG-Agent 因果推理策略扩展示例")
    print("=" * 60)
    
    # 运行经济学示例
    await economic_causal_analysis_example()
    
    print("\n" + "=" * 60)
    
    # 运行社会科学示例
    await social_science_example()
    
    print("\n✨ 因果推理策略扩展示例完成！")


if __name__ == "__main__":
    asyncio.run(main())