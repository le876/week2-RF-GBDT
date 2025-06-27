# 梯度提升决策树(GBDT)生成过程流程图

下面的流程图详细展示了ackley_gbdt项目中梯度提升决策树(GBDT)模型的生成过程，从数据准备到模型评估的完整流程。

## GBDT训练流程

```mermaid
flowchart TD
    A[数据加载与预处理] --> B[特征工程]
    B --> C[数据集划分]
    C --> D[超参数优化]
    
    subgraph 超参数优化过程
        D --> E{选择优化方法}
        E -->|网格搜索| F[GridSearchCV]
        E -->|贝叶斯优化| G[Optuna]
        
        F --> H[交叉验证评估]
        G --> H
        
        H --> I[选择最佳参数]
    end
    
    I --> J[使用最佳参数训练最终模型]
    J --> K[模型评估]
    K --> L[可视化结果]
```

## GBDT模型构建详细流程

```mermaid
flowchart TD
    A[输入数据] --> B[数据预处理]
    B --> C[划分训练集和测试集]
    
    C --> D[GBDT训练]
    
    subgraph GBDT构建过程
        D --> E[初始化模型F0]
        E --> F[迭代训练弱学习器]
        
        subgraph 迭代训练过程
            F --> G[计算负梯度作为残差]
            G --> H[训练决策树拟合残差]
            H --> I[计算叶节点输出值]
            I --> J[更新模型]
            J --> K{是否达到终止条件}
            K -->|否| G
            K -->|是| L[最终模型]
        end
    end
    
    L --> M[预测:所有树的输出相加]
    M --> N[模型评估]
    
    N --> O[计算MSE]
    N --> P[计算R²]
    N --> Q[计算Pearson相关系数]
```

## 单棵决策树训练流程

```mermaid
flowchart TD
    A[残差数据] --> B[决策树训练]
    
    subgraph 决策树构建过程
        B --> C[选择最佳分裂特征]
        
        subgraph 分裂特征选择
            C --> D[计算分裂前的MSE]
            D --> E[尝试每个特征的不同分裂点]
            E --> F[计算分裂后的MSE减少量]
            F --> G[选择MSE减少最大的分裂]
        end
        
        G --> H[分裂节点]
        H --> I[对左右子节点重复分裂过程]
        I --> J{检查停止条件}
        J -->|达到max_depth| K[停止分裂]
        J -->|样本数小于min_samples_split| K
        J -->|继续分裂| I
    end
    
    K --> L[计算叶节点输出值]
    L --> M[返回训练好的决策树]
```

## 贝叶斯优化流程

```mermaid
flowchart TD
    A[定义参数搜索空间] --> B[创建Optuna Study]
    B --> C[开始优化循环]
    
    subgraph 优化循环
        C --> D[生成新的参数组合]
        D --> E[构建GBDT模型]
        E --> F[K折交叉验证]
        
        subgraph 交叉验证过程
            F --> G[训练模型]
            G --> H[验证集预测]
            H --> I[计算Pearson相关系数]
            I --> J[记录性能]
        end
        
        J --> K[更新贝叶斯模型]
        K --> L{达到终止条件?}
        L -->|否| D
        L -->|是| M[选择最佳参数]
    end
    
    M --> N[使用最佳参数训练最终模型]
    N --> O[在测试集上评估]
    O --> P[保存模型和结果]
```

## GBDT中的超参数影响

```mermaid
flowchart TD
    A[GBDT超参数] --> B[弱学习器相关参数]
    A --> C[学习过程相关参数]
    A --> D[正则化相关参数]
    
    subgraph 弱学习器参数
        B --> B1[n_estimators: 弱学习器数量]
        B --> B2[max_depth: 树的最大深度]
        B --> B3[min_samples_split: 分裂所需最小样本数]
        B --> B4[min_samples_leaf: 叶节点最小样本数]
    end
    
    subgraph 学习过程参数
        C --> C1[learning_rate: 学习率]
        C --> C2[subsample: 样本抽样比例]
    end
    
    subgraph 正则化参数
        D --> D1[n_iter_no_change: 早停轮数]
        D --> D2[validation_fraction: 验证集比例]
        D --> D3[tol: 早停容忍度]
    end
    
    B1 --> E[增加可提高模型复杂度]
    B2 --> F[增加可提高拟合能力]
    C1 --> G[减小可提高泛化能力]
    C2 --> H[减小可防止过拟合]
    D1 --> I[增加可防止过拟合]
```

## GBDT预测过程

```mermaid
flowchart TD
    A[输入测试数据] --> B[数据预处理]
    
    subgraph GBDT预测过程
        B --> C[初始预测值F0]
        
        C --> D[树1预测]
        D --> E[树2预测]
        E --> F[树3预测]
        F --> G[...更多树]
        
        G --> H[将所有树的预测值相加]
        H --> I[乘以学习率]
        I --> J[加上初始预测值]
    end
    
    J --> K[最终预测结果]
    K --> L[计算评估指标]
```

## GBDT与随机森林的区别

```mermaid
flowchart TD
    A[集成学习方法] --> B[随机森林]
    A --> C[梯度提升决策树]
    
    subgraph 随机森林特点
        B --> B1[并行训练多棵树]
        B --> B2[每棵树独立构建]
        B --> B3[通过平均减少方差]
        B --> B4[使用Bootstrap抽样]
        B --> B5[随机特征选择]
    end
    
    subgraph GBDT特点
        C --> C1[序列训练多棵树]
        C --> C2[每棵树拟合前一棵树的残差]
        C --> C3[通过加法模型减少偏差]
        C --> C4[使用全部或部分样本]
        C --> C5[使用全部特征]
    end
    
    B3 --> D[适合处理高维数据]
    C3 --> E[适合处理复杂非线性关系]
    
    B1 --> F[训练速度快]
    C1 --> G[训练速度相对慢]
```

## GBDT在ackley_gbdt项目中的参数设置

```mermaid
flowchart TD
    A[ackley_gbdt项目参数] --> B[基本参数]
    A --> C[优化参数范围]
    
    subgraph 基本参数
        B --> B1["n_estimators: 500-2000"]
        B --> B2["learning_rate: 0.05-0.2"]
        B --> B3["max_depth: 4-8"]
        B --> B4["subsample: 0.7-1.0"]
        B --> B5["min_samples_split: 2-8"]
    end
    
    subgraph 固定参数
        C --> C1["n_iter_no_change: 15"]
        C --> C2["validation_fraction: 0.1"]
        C --> C3["tol: 1e-4"]
        C --> C4["random_state: 42"]
    end
    
    B1 --> D[控制模型复杂度]
    B2 --> E[控制学习速度]
    B3 --> F[控制树的复杂度]
    B4 --> G[控制随机性]
    C1 --> H[防止过拟合]
``` 