"""
因果推理策略测试
"""

import pytest
import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.kg_agent.extensions.causal_reasoning import CausalReasoningStrategy


class TestCausalReasoningStrategy:
    """因果推理策略测试类"""
    
    def setup_method(self):
        """测试初始化"""
        self.strategy = CausalReasoningStrategy({
            'max_causal_depth': 3,
            'confidence_threshold': 0.6
        })
    
    @pytest.mark.asyncio
    async def test_basic_causal_analysis(self):
        """测试基本因果分析"""
        task = {
            'entities': ['A', 'B'],
            'hypothesis': {
                'cause': 'A',
                'effect': 'B'
            }
        }
        
        knowledge = {
            'subgraph': {
                'entities': {
                    'A': {'type': 'cause'},
                    'B': {'type': 'effect'}
                },
                'relations': {
                    'A_B': {
                        'type': 'causal',
                        'strength': 0.8
                    }
                }
            }
        }
        
        result = await self.strategy.reason(task, knowledge, None)
        
        assert result['reasoning_type'] == 'causal_reasoning'
        assert 'causal_strength' in result
        assert 'causal_paths' in result
        assert result['confidence'] >= 0.0
    
    @pytest.mark.asyncio
    async def test_intervention_analysis(self):
        """测试干预效果分析"""
        task = {
            'entities': ['X', 'Y'],
            'hypothesis': {
                'cause': 'X',
                'effect': 'Y',
                'intervention': {
                    'variable': 'X',
                    'value': 1.0
                }
            }
        }
        
        knowledge = {
            'subgraph': {
                'entities': {
                    'X': {'type': 'treatment'},
                    'Y': {'type': 'outcome'}
                },
                'relations': {
                    'X_Y': {
                        'type': 'causal',
                        'strength': 0.7
                    }
                }
            }
        }
        
        result = await self.strategy.reason(task, knowledge, None)
        
        intervention = result['intervention_effects']
        assert 'total_effect' in intervention
        assert intervention['target_variable'] == 'X'
    
    @pytest.mark.asyncio
    async def test_counterfactual_analysis(self):
        """测试反事实分析"""
        task = {
            'entities': ['T', 'O'],
            'hypothesis': {
                'cause': 'T',
                'effect': 'O',
                'counterfactual': {
                    'variable': 'T',
                    'original_value': 0,
                    'counterfactual_value': 1
                }
            }
        }
        
        knowledge = {
            'subgraph': {
                'entities': {
                    'T': {'type': 'treatment'},
                    'O': {'type': 'outcome'}
                },
                'relations': {
                    'T_O': {
                        'type': 'causal',
                        'strength': 0.6
                    }
                }
            }
        }
        
        result = await self.strategy.reason(task, knowledge, None)
        
        counterfactual = result['counterfactuals']
        assert 'factual_outcome' in counterfactual
        assert 'counterfactual_outcome' in counterfactual
        assert 'individual_treatment_effect' in counterfactual
    
    @pytest.mark.asyncio
    async def test_confounder_identification(self):
        """测试混杂因素识别"""
        task = {
            'entities': ['X', 'Y', 'Z'],
            'hypothesis': {
                'cause': 'X',
                'effect': 'Y'
            }
        }
        
        knowledge = {
            'subgraph': {
                'entities': {
                    'X': {'type': 'cause'},
                    'Y': {'type': 'effect'},
                    'Z': {'type': 'confounder'}
                },
                'relations': {
                    'Z_X': {
                        'type': 'causal',
                        'strength': 0.5
                    },
                    'Z_Y': {
                        'type': 'causal',
                        'strength': 0.4
                    },
                    'X_Y': {
                        'type': 'causal',
                        'strength': 0.3
                    }
                }
            }
        }
        
        result = await self.strategy.reason(task, knowledge, None)
        
        confounders = result['confounders']
        assert isinstance(confounders, list)
        # 应该识别Z为混杂因素
    
    @pytest.mark.asyncio
    async def test_empty_input_handling(self):
        """测试空输入处理"""
        task = {}
        knowledge = {}
        
        result = await self.strategy.reason(task, knowledge, None)
        
        assert result['confidence'] == 0.0
        assert result['causal_strength']['overall_strength'] == 0.0
    
    @pytest.mark.asyncio
    async def test_complex_causal_chain(self):
        """测试复杂因果链分析"""
        task = {
            'entities': ['A', 'B', 'C', 'D'],
            'hypothesis': {
                'cause': 'A',
                'effect': 'D'
            }
        }
        
        knowledge = {
            'subgraph': {
                'entities': {
                    'A': {}, 'B': {}, 'C': {}, 'D': {}
                },
                'relations': {
                    'A_B': {'type': 'causal', 'strength': 0.8},
                    'B_C': {'type': 'causal', 'strength': 0.7},
                    'C_D': {'type': 'causal', 'strength': 0.6},
                    'A_D': {'type': 'causal', 'strength': 0.3}
                }
            }
        }
        
        result = await self.strategy.reason(task, knowledge, None)
        
        paths = result['causal_paths']
        assert len(paths) > 0
        
        # 应该找到直接路径 A->D 和间接路径 A->B->C->D
        direct_paths = [p for p in paths if p['direct']]
        indirect_paths = [p for p in paths if not p['direct']]
        
        assert len(direct_paths) >= 1
        assert len(indirect_paths) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])