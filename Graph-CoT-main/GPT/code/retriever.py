import glob
import os
import sys
import json
import pickle
from contextlib import nullcontext
from typing import Dict, List
import logging
import math
import re

import faiss
import numpy as np
import torch
from torch.cuda import amp
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
import sentence_transformers

from types import SimpleNamespace as Namespace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from IPython import embed

from bs4 import BeautifulSoup


def clean_html(text):
    """清理HTML标签 - 改进版本"""
    if not isinstance(text, str):
        return str(text) if text is not None else ""

    # 处理不完整的HTML标签
    text = re.sub(r'<[^>]*$', '', text)  # 移除不完整的开始标签
    text = re.sub(r'^[^<]*>', '', text)  # 移除不完整的结束标签

    # 使用BeautifulSoup清理完整的HTML标签
    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text().strip()

    # 如果清理后为空，返回原始文本（去除HTML标签）
    if not cleaned_text:
        cleaned_text = re.sub(r'<[^>]*>', '', text).strip()

    return cleaned_text if cleaned_text else "No title"


NODE_TEXT_KEYS = {'maple': {'paper': ['title'], 'author': ['name'], 'venue': ['name']},
                  'amazon': {'item': ['title'], 'brand': ['name']},
                  'biomedical': {'Anatomy': ['name'], 'Biological_Process': ['name'], 'Cellular_Component': ['name'],
                                 'Compound': ['name'], 'Disease': ['name'], 'Gene': ['name'],
                                 'Molecular_Function': ['name'], 'Pathway': ['name'], 'Pharmacologic_Class': ['name'],
                                 'Side_Effect': ['name'], 'Symptom': ['name']},
                  'legal': {'opinion': ['plain_text'], 'opinion_cluster': ['syllabus'],
                            'docket': ['pacer_case_id', 'case_name'], 'court': ['full_name']},
                  'goodreads': {'book': ['title'], 'author': ['name'], 'publisher': ['name'], 'series': ['title']},
                  'dblp': {'paper': ['title'], 'author': ['name', 'organization'], 'venue': ['name']}
                  }

RELATION_NODE_TYPE_MAP = {
    'maple': {'author': 'author', 'venue': 'venue', 'reference': 'paper', 'cited_by': 'paper', 'paper': 'paper'},
    'amazon': {'also_viewed_item': 'item', 'buy_after_viewing_item': 'item', 'also_bought_item': 'item',
               'bought_together_item': 'item', 'brand': 'brand', 'item': 'item'},
    'biomedical': {'Disease-localizes-Anatomy': ['Anatomy', 'Disease'], 'Anatomy-expresses-Gene': ['Anatomy', 'Gene'],
                   'Anatomy-downregulates-Gene': ['Anatomy', 'Gene'], 'Anatomy-upregulates-Gene': ['Anatomy', 'Gene'],
                   'Gene-participates-Biological Process': ['Biological_Process', 'Gene'],
                   'Gene-participates-Cellular Component': ['Cellular_Component', 'Gene'],
                   'Compound-causes-Side Effect': ['Compound', 'Side_Effect'],
                   'Compound-resembles-Compound': ['Compound'], 'Compound-binds-Gene': ['Compound', 'Gene'],
                   'Compound-downregulates-Gene': ['Compound', 'Gene'],
                   'Compound-palliates-Disease': ['Compound', 'Disease'],
                   'Pharmacologic Class-includes-Compound': ['Compound', 'Pharmacologic_Class'],
                   'Compound-upregulates-Gene': ['Compound', 'Gene'],
                   'Compound-treats-Disease': ['Compound', 'Disease'],
                   'Disease-upregulates-Gene': ['Disease', 'Gene'], 'Disease-downregulates-Gene': ['Disease', 'Gene'],
                   'Disease-associates-Gene': ['Disease', 'Gene'], 'Disease-presents-Symptom': ['Disease', 'Symptom'],
                   'Disease-resembles-Disease': ['Disease'], 'Gene-regulates-Gene': ['Gene'],
                   'Gene-interacts-Gene': ['Gene'],
                   'Gene-participates-Pathway': ['Gene', 'Pathway'],
                   'Gene-participates-Molecular Function': ['Gene', 'Molecular_Function'],
                   'Gene-covaries-Gene': ['Gene']},
    'legal': {'opinion_cluster': 'opinion_cluster', 'reference': 'opinion', 'cited_by': 'opinion', 'opinion': 'opinion',
              'docket': 'docket', 'court': 'court'},
    'goodreads': {'author': 'author', 'publisher': 'publisher', 'series': 'series', 'similar_books': 'book',
                  'book': 'book'},
    'dblp': {'author': 'author', 'venue': 'venue', 'reference': 'paper', 'cited_by': 'paper', 'paper': 'paper'}
}

FEATURE_NODE_TYPE = {'maple': {'paper': ['title'], 'author': ['name'], 'venue': ['name']},
                     'amazon': {'item': ['title', 'price', 'category'], 'brand': ['name']},
                     'biomedical': {'Anatomy': ['name'], 'Biological_Process': ['name'], 'Cellular_Component': ['name'],
                                    'Compound': ['name'], 'Disease': ['name'], 'Gene': ['name'],
                                    'Molecular_Function': ['name'], 'Pathway': ['name'],
                                    'Pharmacologic_Class': ['name'], 'Side_Effect': ['name'], 'Symptom': ['name']},
                     'legal': {'opinion': ['plain_text'], 'opinion_cluster': ['syllabus', 'judges'],
                               'docket': ['pacer_case_id', 'case_name'],
                               'court': ['full_name', 'start_date', 'end_date', 'citation_string']},
                     'goodreads': {
                         'book': ['title', 'popular_shelves', 'genres', 'publication_year', 'num_pages', 'is_ebook',
                                  'language_code', 'format'], 'author': ['name'], 'publisher': ['name'],
                         'series': ['title']},
                     'dblp': {'paper': ['title'], 'author': ['name', 'organization'], 'venue': ['name']}
                     }


class Retriever:

    def __init__(self, args, graph, cache=True, cache_dir=None):
        logger.info("Initializing retriever")

        self.dataset = "amazon"
        self.use_gpu = args.faiss_gpu
        self.node_text_keys = args.node_text_keys
        self.model_name = args.embedder_name
        self.model = sentence_transformers.SentenceTransformer(args.embedder_name)
        self.graph = graph
        self.cache = args.embed_cache
        self.cache_dir = args.embed_cache_dir

        self.reset()

    def reset(self):
        # 确保缓存文件夹存在
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            logger.info(f"Created cache directory: {self.cache_dir}")

        docs, ids, meta_type = self.process_graph()
        save_model_name = self.model_name.split('/')[-1]

        if self.cache and os.path.isfile(os.path.join(self.cache_dir, f'cache-{save_model_name}.pkl')):
            embeds, self.doc_lookup, self.doc_type = pickle.load(
                open(os.path.join(self.cache_dir, f'cache-{save_model_name}.pkl'), 'rb'))
            assert self.doc_lookup == ids
            assert self.doc_type == meta_type
        else:
            embeds = self.multi_gpu_infer(docs)
            self.doc_lookup = ids
            self.doc_type = meta_type
            pickle.dump([embeds, ids, meta_type],
                        open(os.path.join(self.cache_dir, f'cache-{save_model_name}.pkl'), 'wb'))

        self.init_index_and_add(embeds)

    def process_graph(self):
        docs = []
        ids = []
        meta_type = []

        for node_type_key in self.graph.keys():
            node_type = node_type_key.split('_nodes')[0]
            logger.info(f'loading text for {node_type}')
            for nid in tqdm(self.graph[node_type_key]):
                tmp_string = ''
                try:
                    for k in self.node_text_keys[node_type]:
                        if k in self.graph[node_type_key][nid]['features']:
                            vv = self.graph[node_type_key][nid]['features'][k]

                            # 改进的值处理
                            if isinstance(vv, str):
                                if k == 'title':
                                    vv = clean_html(vv)
                                if vv.strip():  # 只有非空字符串才添加
                                    tmp_string += f"{k}: {vv}. "
                            elif isinstance(vv, (int, float)) and not math.isnan(vv):
                                tmp_string += f"{k}: {str(vv)}. "
                            elif isinstance(vv, list) and vv:
                                tmp_string += f"{k}: {', '.join(str(x) for x in vv)}. "

                    # 如果没有有效内容，添加基本信息
                    if not tmp_string.strip():
                        tmp_string = f"Node ID: {nid}. Type: {node_type}. "

                except Exception as e:
                    logger.warning(f"Error processing node {nid}: {e}")
                    tmp_string = f"Node ID: {nid}. Type: {node_type}. "

                docs.append(tmp_string)
                ids.append(nid)
                meta_type.append(node_type)

                # Debug: 检查生成的文本
                if len(tmp_string) < 20:
                    logger.info(f"Short context for {nid}: {tmp_string}")

        logger.info(f"Processed {len(docs)} nodes")
        return docs, ids, meta_type

    def multi_gpu_infer(self, docs):
        pool = self.model.start_multi_process_pool()
        embeds = self.model.encode_multi_process(docs, pool)
        return embeds

    def _initialize_faiss_index(self, dim: int):
        self.index = None
        cpu_index = faiss.IndexFlatIP(dim)
        self.index = cpu_index

    def _move_index_to_gpu(self):
        logger.info("Moving index to GPU")
        ngpu = faiss.get_num_gpus()
        gpu_resources = []
        for i in range(ngpu):
            res = faiss.StandardGpuResources()
            gpu_resources.append(res)
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.usePrecomputed = False
        vres = faiss.GpuResourcesVector()
        vdev = faiss.Int32Vector()
        for i in range(0, ngpu):
            vdev.push_back(i)
            vres.push_back(gpu_resources[i])
        self.index = faiss.index_cpu_to_gpu_multiple(vres, vdev, self.index, co)

    def init_index_and_add(self, embeds):
        logger.info("Initialize the index...")
        dim = embeds.shape[1]
        self._initialize_faiss_index(dim)
        self.index.add(embeds)

        if self.use_gpu:
            self._move_index_to_gpu()

    def search_single(self, query, hop=1, topk=10):
        if self.index is None:
            raise ValueError("Index is not initialized")

        query_embed = self.model.encode(query, show_progress_bar=False)

        D, I = self.index.search(query_embed[None, :], topk)

        # 获取所有topk结果的信息
        top_indices = np.array(self.doc_lookup)[I].tolist()[0]
        top_types = np.array(self.doc_type)[I].tolist()[0]
        top_scores = D.tolist()[0]

        logger.info(f"Top {len(top_indices)} results for query '{query}':")
        for i, (idx, node_type, score) in enumerate(zip(top_indices, top_types, top_scores)):
            logger.info(f"  {i + 1}. Node: {idx}, Type: {node_type}, Score: {score:.4f}")

        # 使用最佳匹配
        original_indice = top_indices[0]
        original_type = top_types[0]

        try:
            if hop == 1:
                context = self.one_hop(original_type, original_indice)
            elif hop == 2:
                context = self.two_hop(original_type, original_indice)
            elif hop == 0:
                context = self.zero_hop(original_type, original_indice)
            else:
                raise ValueError('Ego graph should be 0-hop, 1-hop or 2-hop.')
        except Exception as e:
            logger.error(f"Error generating context: {e}")
            context = f"Error: Could not generate context for node {original_indice} of type {original_type}"

        # 保存结果
        result = {
            'query': query,
            'context': context,
            'top_matches': [
                {'node_id': idx, 'node_type': nt, 'score': float(score)}
                for idx, nt, score in zip(top_indices[:5], top_types[:5], top_scores[:5])
            ]
        }

        result_filename = '/Users/yehaoran/Desktop/KGAgentEcno/Graph-CoT-main/GPT/results/GPT_retriever_results.json'
        with open(result_filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        return context

    def linearize_feature(self, node_type, node_indice):
        """改进的特征线性化方法"""
        text = ''
        try:
            node_features = self.graph[f'{node_type}_nodes'][node_indice]['features']

            for f_name in node_features:
                if f_name in FEATURE_NODE_TYPE[self.dataset][node_type]:
                    val = node_features[f_name]

                    # 特殊处理title字段的HTML
                    if f_name == 'title' and isinstance(val, str):
                        val = clean_html(val)

                    # 处理空值
                    if isinstance(val, str):
                        if val.strip():  # 非空字符串
                            text += f"{f_name}: {val}. "
                        else:
                            text += f"{f_name}: No {f_name}. "
                    elif isinstance(val, (int, float)):
                        if not math.isnan(val):
                            text += f"{f_name}: {str(val)}. "
                        else:
                            text += f"{f_name}: No {f_name}. "
                    elif isinstance(val, list):
                        if val:  # 非空列表
                            text += f"{f_name}: {', '.join(str(x) for x in val)}. "
                        else:
                            text += f"{f_name}: No {f_name}. "
                    else:
                        text += f"{f_name}: {str(val)}. "

        except Exception as e:
            logger.error(f"Error linearizing features for {node_type} {node_indice}: {e}")
            text = f"Node {node_indice} of type {node_type}. "

        return text

    def zero_hop(self, node_type, node_indice):
        context = 'Center node: '
        context += self.linearize_feature(node_type, node_indice)
        return context

    def one_hop(self, node_type, node_indice, sample_n=20):
        context = 'Center node: '
        context += self.linearize_feature(node_type, node_indice)

        try:
            node_neighbors = self.graph[f'{node_type}_nodes'][node_indice]['neighbors']

            for neighbor_type in node_neighbors:
                context += f"\n{neighbor_type}: "
                neighbor_count = 0

                for nid in node_neighbors[neighbor_type][:sample_n]:
                    try:
                        if isinstance(RELATION_NODE_TYPE_MAP[self.dataset][neighbor_type], str):
                            neighbor_node_type = RELATION_NODE_TYPE_MAP[self.dataset][neighbor_type]
                            neighbor_features = self.linearize_feature(neighbor_node_type, nid)
                            if neighbor_features.strip():
                                context += neighbor_features
                                neighbor_count += 1
                        elif isinstance(RELATION_NODE_TYPE_MAP[self.dataset][neighbor_type], list):
                            for ntt in RELATION_NODE_TYPE_MAP[self.dataset][neighbor_type]:
                                try:
                                    neighbor_features = self.linearize_feature(ntt, nid)
                                    if neighbor_features.strip():
                                        context += neighbor_features
                                        neighbor_count += 1
                                except:
                                    continue
                    except Exception as e:
                        logger.warning(f"Error processing neighbor {nid}: {e}")
                        continue

                if neighbor_count == 0:
                    context += "No valid neighbors found. "

        except Exception as e:
            logger.error(f"Error in one_hop for {node_type} {node_indice}: {e}")
            context += f"Error accessing neighbors: {e}"

        return context

    def two_hop(self, node_type, node_indice, sample_n=20):
        context = 'Center node: '
        context += self.linearize_feature(node_type, node_indice)

        try:
            node_neighbors = self.graph[f'{node_type}_nodes'][node_indice]['neighbors']

            for neighbor_type in node_neighbors:
                context += f"\n{neighbor_type}: "

                for nid in node_neighbors[neighbor_type][:sample_n]:
                    try:
                        neighbor_context = self.one_hop(neighbor_type, nid, sample_n=5)  # 减少二跳的采样
                        context += f'[{neighbor_context}]. '
                    except Exception as e:
                        logger.warning(f"Error in two_hop for neighbor {nid}: {e}")
                        continue

        except Exception as e:
            logger.error(f"Error in two_hop for {node_type} {node_indice}: {e}")
            context += f"Error accessing two-hop neighbors: {e}"

        return context


if __name__ == '__main__':
    # ====== 构造 args ======
    args = Namespace(
        faiss_gpu=False,
        node_text_keys=NODE_TEXT_KEYS['amazon'],
        embedder_name="sentence-transformers/all-mpnet-base-v2",
        embed_cache=True,
        embed_cache_dir="./cache"
    )

    graph_dir = "/Users/yehaoran/Desktop/KGAgentEcno/Graph-CoT-main/data/processed_data/amazon/magazine_graph.json"
    query = "quantum physics and machine learning"

    graph = json.load(open(graph_dir))
    node_retriever = Retriever(args, graph)

    # 执行查询
    context = node_retriever.search_single(query, 1)
    print("=" * 50)
    print("RETRIEVAL RESULT:")
    print("=" * 50)
    print(context)

    # 创建文件夹并保存结果
    results_folder = '/Users/yehaoran/Desktop/KGAgentEcno/Graph-CoT-main/GPT/results'
    os.makedirs(results_folder, exist_ok=True)

    result_filename = os.path.join(results_folder, 'GPT_retriever_results.json')
    result = {'query': query, 'context': context}

    with open(result_filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"\nResults saved to {result_filename}")