# 关于如何向Graph-CoT文件夹增加更改
由于Graph-CoT为其他仓库clone，所以需要在Graph-CoT文件夹下进行更改，然后提交到Graph-CoT仓库，最后在父仓库中更新对子模块的引用。

## 1. 进入子模块目录
cd /Users/liangzhang/Desktop/project/Agent-KG-ecno/Graph-CoT

## 2. 现在你可以在子模块中添加和提交文件
git add data/processed_data/amazon/amazon_magzine_graph.json

git commit -m "Add amazon magazine graph data"

git push origin main  # 假设子模块也使用 main 分支

## 3. 返回父仓库
cd ..

## 4. 在父仓库中更新对子模块的引用
git add Graph-CoT

git commit -m "Update Graph-CoT submodule reference"

git push origin main