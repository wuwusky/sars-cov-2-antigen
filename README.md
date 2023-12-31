# sars-cov-2-antigen  
https://tianchi.aliyun.com/forum/post/471967  

初赛阶段提交代码描述：
1.模型无需大规模预训练的蛋白质类模型，直接设计的基于残差模块的一维神经网路模型，分别对抗体和抗原进行embedding后合并然后对亲和力和活性任务进行预测，能兼容较长序列（~1500）,设计了空洞卷积金字塔结构模块，增加网络感受野，获取更长的上下文关系；  
2.模型流程目前只用到了序列本身的信息（fasta），还未加入结构信息进行进一步的特征embedding    
3.基于提交的这一版本，后续结合了transfomer进一步提升模型对序列上下文的特征提取能力，仍然无需预训练，并加入了残基的化学性质信息（亲水，疏水，酸碱性）,仍然是轻量型模型，利于后续基于预测模型的蛋白序列denovo生成或设计    
4.任务标签由1-5转换为[00000]-[11111]，将分类任务转换为回归任务，预测值相加得到近似的1-5的等级，能缓解样本分布不平衡的问题，soft-label相比原始标签能降低拟合风险   
5.加入重采样平衡样本类别平衡问题    

PS:训练代码因为迭代跟当前提交模型版本有出入，训练超参和本地验证评估方式有变更

=========================================================初赛/复赛分割线===================================================================   
复赛提交代码和模型分为两个版本s1和s2    
s1版本：基于序列信息（轻链，重链，CDRH3和CDRL3）和抗原序列进行预测，对亲和力和活性进行预测（初赛版本），线上验证测试指标为：0.27    
s2版本：基于序列信息（轻链，重链）和更多抗原序列进行预测，在训练阶段加入了序列结构先验进行约束模型，对亲和力和活性进行预测（复赛结构版本），本地数据（30%）验证指标为0.35±0.02，线上验证测试指标暂为：0.24

s1版本模型代码描述见初赛阶段提交内容描述。    
验证指令：bash test.sh
训练指令：python train_a_s1.py  /  python train_n_s1.py  /  

s2版本建模描述：    
1.模型设计部分  
测试阶段：重链，轻链以及抗原序列和各自的残基亲疏水性酸碱性信息序列经固定token编码后，分别由空洞卷积金字塔模块进行特征embedding；再经由4层transformer encoder 模块进行上下文的特征编码；之后利用一维残差卷积网络模块进一步的对特征进行编码并完成下采样，分别得到抗原，重链和轻链三个特征信息进行concatenate整合；再分别经亲和力和活性的预测头完成结果预测。    
训练阶段：
训练过程采用同源数据异步多任务学习。    
①在亲和力和活性标签信息的基础上，对比初赛版本，不再使用抗体链的CDR3区域作为输入，而是使用数据中的完整重链和轻链序列来预测各自CDR3区域的起始和终止位置来进一步约束模型的编码特征（编码得到的特征同时用来预测CDR3区域的位置）     
②基于赛题中的pdb文件，提取pdb中每条链的结构信息来制作单链的残基（CA原子）距离关系矩阵，对模型编码部分进一步约束（编码模块提取特征后用来预测接触图）

验证指令：bash test_s2.py   
训练指令：python train_a_split_mt_s2.py  /  python train_n_split_mt_s2.py   


=========================================================复赛/决赛分割线===================================================================   
新增s3版本模型和代码

s3版本建模描述
1.模型设计部分  
测试阶段：  
重链，轻链以及抗原序列利用训练好的蛋白质结构预测模型(OmegaFold)对特征进行编码形成拓扑图和边的特征信息，再经图卷积(GCN)模块进一步整合，得到亲和力和活性的结果预测  
训练阶段：  
训练过程目前采用基础的单任务约束，利用亲和力和活性的标签分别约束模型进行监督训练  

验证指令：bash test_s3.py   
训练指令：python train_a_split_mt_s3.py  /  python train_n_split_mt_s3.py   

===================================================================   
环境依赖：pytorch，pandas，tqdm，scipy, sklearn, numpy

