# Graph-R1 数据集

Graph-R1 论文中用到 6 个数据集的**构建背景、数据规模、问题特点、官方评估指标与典型样例**

---

### 1️⃣ 2WikiMultiHopQA

**类型**：多跳问答（2-hop 为主）

**规模**：训练 167 k / 验证 15 k / 测试 15 k

**来源**：维基百科摘要 + Wikidata 三元组

**构建方式**

- 先用逻辑规则在 Wikidata 上生成“桥接实体”，再调用众包写成自然语言问题，确保必须**跨两篇维基文档**才能回答 。
- 每条样本额外给出 **Wikidata 三元组路径**（头实体-属性-尾实体）作为“结构化解释”，方便做可解释性实验 。

**问题类别**

1. 比较型（Compare）：先搭桥再比较属性
2. 组合型（Compose）：链式关系
3. 推理型（Infer）：隐含属性推断

**评估指标**

- 答案 F1 / EM
- 解释召回：三元组路径命中率

**典型样例**

> Q：Which film directed by the person who directed “In Memory of Sergo Ordzhonikidze” has the earlier release date?
> 
> 
> 需跳 1 找导演 → 跳 2 比较同导演多部影片年份。
> 

**难度特点**

单跳 BERT 基线仅 38 F1，远低于 HotpotQA（55 F1），说明“伪多跳”比例低 。

---

### 2️⃣ HotpotQA

**类型**：多跳问答（2-hop）

**规模**：训练 113 k / 验证 7 k / 测试 7 k

**来源**：英文维基百科导语段落

**构建方式**

- 自动抽取“超链接段落对”，众包标注问题 + **支持句（supporting facts）** 句子级黄金证据 。
- 提供 **Distractor** 与 **Full-wiki** 两种设置：
– Distractor：10 段干扰段，保证黄金段在内；
– Full-wiki：仅给问题，模型需先检索再答。

**问题类别**

- Bridge：先找桥实体再问属性
- Comparison：两个实体就同一属性比较

**评估指标**

- Answer F1 / EM
- Supporting Fact F1 / EM（可解释性）
- Joint F1（两项乘积）

**典型样例**

> Q：Are both directors of “Film A” and “Film B” from the same country?
> 
> 
> 需分别定位两部影片导演 → 再比较国籍。
> 

**难度特点**

- 支持句标注让模型无法靠“ shortcuts”猜答案；
- 开放域设置对检索器挑战大，官方基线 F1≈61。

---

### 3️⃣ MuSiQue

**类型**：多跳问答（2-4 hop）

**规模**：训练 2.2 M 子问题 → 组合成 340 k 多跳 / 验证 20 k / 测试 20 k

**来源**：维基段落

**构建方式**

- **Bottom-up**：先写大量单跳问题，再按语义合成多跳链，确保**无法单跳回答** 。
- 引入**反捷径机制**：把答案文本或同义词从问题里抹掉，防止关键词匹配。

**评估指标**

- Answer F1 / EM
- 跳数准确率（按 2/3/4 hop 分别报告）

**典型样例**

> Q：What is the birth year of the spouse of the director of the 2009 film “X”?
> 
> 
> 需 3 跳：影片→导演→配偶→出生年。
> 

**难度特点**

- 平均需 2.97 跳，是目前公开**跳数最深**的多跳集；
- 单跳模型 F1≈25，显著低于 HotpotQA。

---

### 4️⃣ Natural Questions (NQ)

**类型**：单跳问答

**规模**：训练 307 k / 验证 8 k / 测试 8 k

**来源**：真实 Google 搜索查询 + 维基百科对应段落

**构建方式**

- 搜索日志脱敏后，由标注员给出**长答案**（段落）与**短答案**（实体或短语）。

**评估指标**

- 短答案 F1 / EM（主流 RAG 采用）
- 长答案 F1（生成式任务偶尔用）

**典型样例**

> Q：how tall is the eiffel tower
> 
> 
> 短答案：330 feet
> 

**难度特点**

- 问题为真实用户 query，分布更贴近线上；
- 单跳即可定位，常被用作**RAG 单跳基线**。

---

### 5️⃣ PopQA

**类型**：开放域单跳（实体中心）

**规模**：训练 13 k / 验证 1.4 k / 测试 1.4 k

**来源**：2021-12 维基 dump + 实体流行度过滤

**构建方式**

- 取 Wikidata 高频实体（电影、歌手、体育明星），用模板生成“流行文化”问题。
- 实体出现频率**低于 NQ**，更考验检索器的实体链接能力。

**评估指标**

- Answer F1 / EM

**典型样例**

> Q：Who is the lead vocalist of the band that released the album “After Hours”?
> 
> 
> 需先定位乐队 The Weeknd → 再确认主唱即本人。
> 

**难度特点**

- 问题短、实体新，测试**长尾实体召回**；
- 单跳模型 F1≈55，低于 NQ（68 F1）。

---

### 6️⃣ TriviaQA

**类型**：开放域常识问答

**规模**：训练 96 k / 验证 11 k / 测试 11 k

**来源**： trivia 网站（非维基）+ 远监督对齐

**构建方式**

- 先收集 trivia 问答题，再用搜索引擎回标维基/网络文档，形成**远监督证据**。

**评估指标**

- Answer F1 / EM
- 文档级检索召回（可选）

**典型样例**

> Q：Which planet has the most moons?
> 
> 
> 答案：Saturn（截至 2025 已确认 146 颗卫星）
> 

**难度特点**

- 问题偏向**常识&冷门知识**，对参数记忆要求高；
- 远监督证据含噪声，考验**去噪检索**能力。

---

### 快速选型建议

| 任务目标 | 推荐数据集 |
| --- | --- |
| 纯单跳 RAG 基线 | NQ → PopQA |
| 多跳检索+解释 | HotpotQA（有支持句） |
| 更深多跳/反捷径 | MuSiQue |
| 真实用户 query | NQ |
| 常识&闭卷测试 | TriviaQA |
| 结构化解释 | 2WikiMultiHopQA |

# 实例和数据集链接

---

### 1️⃣ 2WikiMultiHopQA

官方主页：[https://github.com/Alab-NII/2wikimultihop](https://github.com/Alab-NII/2wikimultihop)

在线浏览器：✅ 支持（含三元组证据链）

| 示例 ID | 问题 | 答案 | 证据链（Wikidata 三元组） |
| --- | --- | --- | --- |
| 2Wiki-train-0001 | **Who is the grandfather of the husband of Victoria, Crown Princess of Sweden?** | Carl XVI Gustaf | Victoria → spouse → Daniel → father → Carl XVI Gustaf |
| 2Wiki-train-0012 | **Which film directed by the director of “In Memory of Sergo Ordzhonikidze” was released earlier?** | A Sixth Part of the World | Dziga Vertov → directed → both films；比较上映年份 |
| 2Wiki-dev-0055 | **The country of which the father of the current King of Spain is from?** | Greece | Juan Carlos I → father → Infante Juan → country → Greece |

---

### 2️⃣ HotpotQA

官方主页：[https://hotpotqa.github.io](https://hotpotqa.github.io/)

在线浏览器：✅ 支持（含**绿色支撑句**高亮）

| 示例 ID | 问题 | 答案 | 支撑句（绿色高亮） |
| --- | --- | --- | --- |
| train-easy-0001 | **Are both the director of “Film A” and the director of “Film B” from the same country?** | yes | “Film A was directed by **X**…”; “Film B was directed by **X**…; **X** is from **France**.” |
| train-medium-0123 | **Which university did the MVP of the 2015 Diamond Head Classic play for?** | Oklahoma | “Buddy Hield was named MVP…”; “Hield played college basketball for **Oklahoma**.” |
| dev-distractor-0456 | **Who is the spouse of the director of “In Memory of Sergo Ordzhonikidze”?** | Yelizaveta Svilova | “**Dziga Vertov** directed…”; “Vertov’s spouse was **Yelizaveta Svilova**.” |

🔗 在线样例直达：https://hotpotqa.github.io/explorer.html （输入D 即可看到绿色支撑句）

---

### 3️⃣ MuSiQue

官方主页：[https://github.com/stonybrooknlp/musique](https://github.com/stonybrooknlp/musique)

在线浏览器：❌ 无，但 GitHub 提供**JSON 样例下载**（含子问题分解）

| 示例 ID | 问题 | 答案 | 单跳子链（2→3→4 hop） |
| --- | --- | --- | --- |
| musique-train-0001 | **What is the birth year of the spouse of the director of the 2009 film “Veronica”?** | 1888 | 影片→导演→配偶→出生年（3-hop） |
| musique-train-0077 | **How many days after the release of film X was the director’s spouse born?** | 14235 | 影片→上映日；导演→配偶→出生日；日期差（4-hop） |
| musique-dev-0199 | **In which country was the father of the actor who played “Y” in film Z born?** | France | 影片→演员→父亲→出生国（3-hop） |

---

### 4️⃣ Natural Questions (NQ)

官方主页：[https://ai.google.com/research/NaturalQuestions](https://ai.google.com/research/NaturalQuestions)

在线浏览器：✅ Google 官方可视化工具

| 示例 ID | 问题 | 短答案 | 长答案段落 |
| --- | --- | --- | --- |
| train-0001 | **where is the eiffel tower located** | Paris, France | “The Eiffel Tower is located on the Champ de Mars in **Paris, France**…” |
| train-0456 | **who wrote the book the old man and the sea** | Ernest Hemingway | “**The Old Man and the Sea** is a novel written by **Ernest Hemingway**…” |
| dev-0789 | **what is the capital of the country where victoria falls is located** | Livingstone (Zambia) / Victoria Falls (Zimbabwe) | 需单跳定位“Victoria Falls”所在国 → 再定位首都 |

🔗 在线可视化：[https://ai.google.com/research/NaturalQuestions/visualization](https://ai.google.com/research/NaturalQuestions/visualization) （输入 ID 即可看到用户真实查询与维基段落）

---

### 5️⃣ PopQA

官方主页：[https://github.com/AlexMall/PopQA](https://github.com/AlexTMallen/adaptive-retrieval)

在线浏览器：❌ 无，但 GitHub 提供**TSV 样例文件**（含实体流行度分数）

| 示例 ID | 问题 | 答案 | 实体流行度（月均维基浏览量） |
| --- | --- | --- | --- |
| pop-train-0001 | **Who is the lead vocalist of the band that released the album “After Hours”?** | The Weeknd | 1.9 M |
| pop-train-0123 | **Who is the mother of the character played by Emma Watson in the Harry Potter films?** | Molly Weasley | 420 k |
| dev-0456 | **Which actor portrayed the main character in the movie “Inception”?** | Leonardo DiCaprio | 2.1 M |

---

### 6️⃣ TriviaQA

官方主页：[http://nlp.cs.washington.edu/triviaqa/](http://nlp.cs.washington.edu/triviaqa/)

在线浏览器：❌ 无，但提供**远监督文档链接**（可下载）

| 示例 ID | 问题 | 答案 | 远监督文档（含噪声） |
| --- | --- | --- | --- |
| train-0001 | **Which planet has the most moons?** | Saturn | 来自 [space.com](http://space.com/) 文章，需去噪 |
| train-0234 | **Who painted the ceiling of the Sistine Chapel?** | Michelangelo | 维基+艺术博客多篇文章 |
| dev-0567 | **What is the chemical symbol for gold?** | Au | 常识型，文档分散 |

---

### 尝试获取数据集的代码

```bash
# 0. 建目录
mkdir -p datasets/{2wiki,hotpot,musique,nq,popqa,triviaqa}

# 1. 2WikiMultiHopQA
wget -P datasets/2wiki <http://qang.cs.ucl.ac.uk/2wikimultihopqa/dataset.tar.gz>

# 2. HotpotQA
wget -P datasets/hotpot <https://dl.fbaipublicfiles.com/hotpotqa/hotpot_train_v1.1.json> \\
                      <https://dl.fbaipublicfiles.com/hotpotqa/hotpot_dev_distractor_v1.json>

# 3. MuSiQue
wget -P datasets/musique <https://github.com/allenai/musique/releases/download/v1.0/musique_v1.0.tar.gz>

# 4. NQ
gsutil -m cp -R gs://natural_questions/v1.0 datasets/nq

# 5. PopQA
wget -P datasets/popqa <https://raw.githubusercontent.com/AlexMall/PopQA/main/data/popqa.tsv>

# 6. TriviaQA
wget -P datasets/triviaqa <http://nlp.cs.washington.edu/triviaqa/data/triviaqa-rc.tar.gz>

```

---