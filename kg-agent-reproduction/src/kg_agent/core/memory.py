import json
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pickle
import hashlib

class MemoryStore:
    """记忆存储"""
    
    def __init__(self, storage_type: str = "local"):
        self.storage_type = storage_type
        self.episodic_memory = []
        self.semantic_memory = {}
        self.procedural_memory = {}
        
    def store_episode(self, episode: Dict[str, Any]):
        """存储事件记忆"""
        episode['timestamp'] = datetime.now().isoformat()
        episode['id'] = self._generate_id(episode)
        self.episodic_memory.append(episode)
        
        # 保持记忆大小限制
        max_size = 1000
        if len(self.episodic_memory) > max_size:
            self.episodic_memory = self.episodic_memory[-max_size:]
    
    def store_semantic(self, key: str, value: Any):
        """存储语义记忆"""
        self.semantic_memory[key] = {
            'value': value,
            'timestamp': datetime.now().isoformat(),
            'access_count': 0
        }
    
    def store_procedural(self, task_type: str, procedure: Dict[str, Any]):
        """存储程序记忆"""
        if task_type not in self.procedural_memory:
            self.procedural_memory[task_type] = []
        
        procedure['timestamp'] = datetime.now().isoformat()
        procedure['success_rate'] = 0.0
        self.procedural_memory[task_type].append(procedure)
    
    def retrieve_episodic(self, query: Dict[str, Any], 
                         limit: int = 10) -> List[Dict[str, Any]]:
        """检索事件记忆"""
        # 基于相似度检索
        similarities = []
        
        for episode in self.episodic_memory:
            similarity = self._calculate_similarity(query, episode)
            similarities.append((similarity, episode))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        return [episode for _, episode in similarities[:limit]]
    
    def retrieve_semantic(self, key: str) -> Optional[Any]:
        """检索语义记忆"""
        if key in self.semantic_memory:
            self.semantic_memory[key]['access_count'] += 1
            return self.semantic_memory[key]['value']
        return None
    
    def retrieve_procedural(self, task_type: str) -> List[Dict[str, Any]]:
        """检索程序记忆"""
        return self.procedural_memory.get(task_type, [])
    
    def _generate_id(self, item: Dict[str, Any]) -> str:
        """生成唯一ID"""
        content = json.dumps(item, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    def _calculate_similarity(self, query: Dict[str, Any], 
                            episode: Dict[str, Any]) -> float:
        """计算相似度"""
        # 简化的相似度计算
        query_str = json.dumps(query)
        episode_str = json.dumps(episode)
        
        # 使用Jaccard相似度
        query_set = set(query_str.split())
        episode_set = set(episode_str.split())
        
        intersection = len(query_set.intersection(episode_set))
        union = len(query_set.union(episode_set))
        
        return intersection / union if union > 0 else 0.0

class MemorySystem:
    """记忆系统"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.short_term = MemoryStore("short_term")
        self.long_term = MemoryStore("long_term")
        self.working_memory = {}
        
        # 记忆参数
        self.decay_rate = config.get('decay_rate', 0.1)
        self.reinforcement_threshold = config.get('reinforcement_threshold', 3)
    
    async def store_experience(self, task: Dict[str, Any], result: Dict[str, Any]):
        """存储经验"""
        experience = {
            'task': task,
            'result': result,
            'success': result.get('success', False),
            'confidence': result.get('confidence', 0.0)
        }
        
        # 存储到短期记忆
        self.short_term.store_episode(experience)
        
        # 如果成功且置信度高，存储到长期记忆
        if result.get('success', False) and result.get('confidence', 0) > 0.8:
            self.long_term.store_episode(experience)
        
        # 更新程序记忆
        task_type = task.get('task_type', 'unknown')
        procedure = {
            'steps': result.get('reasoning_path', []),
            'success': result.get('success', False),
            'confidence': result.get('confidence', 0.0)
        }
        self.long_term.store_procedural(task_type, procedure)
    
    async def find_similar_cases(self, task: Dict[str, Any], 
                               similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """查找相似案例"""
        query = {
            'task_type': task.get('task_type'),
            'entities': task.get('entities', []),
            'relations': task.get('relations', [])
        }
        
        # 从长期记忆中检索
        similar_cases = self.long_term.retrieve_episodic(query)
        
        # 过滤相似度
        filtered_cases = []
        for case in similar_cases:
            similarity = self._calculate_task_similarity(task, case['task'])
            if similarity >= similarity_threshold:
                filtered_cases.append(case)
        
        return filtered_cases
    
    def _calculate_task_similarity(self, task1: Dict[str, Any], 
                                 task2: Dict[str, Any]) -> float:
        """计算任务相似度"""
        # 基于任务类型、实体和关系的相似度计算
        type_similarity = 1.0 if task1.get('task_type') == task2.get('task_type') else 0.0
        
        entities1 = set(task1.get('entities', []))
        entities2 = set(task2.get('entities', []))
        
        if not entities1 or not entities2:
            entity_similarity = 0.0
        else:
            intersection = len(entities1.intersection(entities2))
            union = len(entities1.union(entities2))
            entity_similarity = intersection / union
        
        return (type_similarity + entity_similarity) / 2
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆统计"""
        return {
            'short_term_size': len(self.short_term.episodic_memory),
            'long_term_size': len(self.long_term.episodic_memory),
            'semantic_memory_size': len(self.long_term.semantic_memory),
            'procedural_memory_size': len(self.long_term.procedural_memory)
        }