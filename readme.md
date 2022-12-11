# 基于层次注意力机制的源代码迁移模型

在软件迁移过程中，开发人员需要使用目标编程语言重新编写项目代码，该过程费时且容易出错。为此，研究人员尝试研究代码自动迁移技术来解决这一问题，期望在不改变程序语义的情况下，将源代码转换成目标代码，从而减轻开发人员的负担。神经机器翻译（Neural Machine Translation，NMT）模型可以帮助实现源代码语义的自动迁移，但这些模型直接翻译代码文本至目标代码，忽略了代码的结构特征，迁移较长代码时效果不佳。

本文在词符注意力机制的基础上，引入语句注意力机制，构建了一种基于层次注意力机制的源代码迁移模型（Hierarchical Pointer-Generator Networks，HPGN）。其在迁移过程中通过关注代码语句的语法和语义，从而进一步提升迁移代码的语义一致性。

目前正在调整项目结构，部分代码正在更改中
```
HPGN
│  hpgn_main.py                  层次指针生成网络——训练
│  metric_result_analysis.py     指标分析
│  pgn_main.py                   指针生成网络——训练
│  readme.md
├─dataset
│  └─Csharp_Java      数据、模型、分词器、参考输出
│      ├─dataset        预处理数据集和单词表
│      ├─model          模型以及输出样例
│      ├─trained-model  本实验模型的输出样例
│      └─Tokenizers     分词器
├─transformer_result    transformer输出结果
├─evaluator         bleu、Codebleu
│  └─CodeBLEU
└─Network           指针生成网络、层次指针生成网络
  
```
# 运行环境
python=3.9  
tensorflow=2.7

# 评测指标 与 数据集
- BLEU：计算生成的序列和参考序列的n-gram重叠率，并返回0到100%之间的分值。BLEU值越高，表示生成的序列越接近参考序列
- 精确匹配（Exact Match，EM）：评测预测输出和参考输出是否完全一致
- CodeBLEU：采用BLEU的n-gram匹配算法，并通过代码解析工具引入了抽象语法树和数据流匹配算法。CodeBLEU根据代码的文本、语法和语义来评估代码，给出0到100%之间的分数。CodeBLEU值越高，代码生成的质量越高

数据集：基于真实项目的(Java-C#)数据集[CodeTrans](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/code-to-code-trans)

# 实验结果

HPGN 隐藏层-64维，网络-1层

## Java到C#
| 模型         | BLEU  | EM   | CodeBLEU |
| ------------ | ----- | ---- | -------- |
| Naive        | 18.54 | 0.0  | -        |
| PBSMT        | 43.53 | 12.5 | 42.71    |
| tree-to-tree | 36.34 | 3.4  | 42.13    |
| Transformer  | 58.53 | 34.4 | 64.20    |
| Resnet-HPGN  | 58.74 | 19.5 | 63.08    |
| Gate-HPGN    | 61.40 | 27.3 | 64.93    |
| Base-HPGN    | 59.79 | 20.7 | 63.88    |
| 指针生成网络 | 26.18 | 13.8 | 43.87    |
## C#到Java
| 模型         | BLEU  | EM   | CodeBLEU |
| ------------ | ----- | ---- | -------- |
| Naive        | 18.69 | 0.0  | -        |
| PBSMT        | 40.06 | 16.1 | 43.48    |
| tree-to-tree | 32.09 | 4.4  | 43.86    |
| Transformer  | 52.87 | 34.7 | 58.56    |
| Resnet-HPGN  | 60.28 | 29.5 | 64.20    |
| Gate-HPGN    | 60.95 | 32.1 | 64.62    |
| Base-HPGN    | 58.26 | 30.0 | 62.35    |
| 指针生成网络 | 27.84 | 20.5 | 44.88    |

# 人工评分
## 指标
<table>
    <tr>
        <th rowspan=2>分值</th>
        <th colspan=2>评价指标</th>
    </tr>
    <tr>
        <th>语法</th>
        <th>语义</th>
    </tr>
    <tr>
        <td>5</td>
        <td>没有语法错误</td>
        <td>语义一致</td>
    </tr>
    <tr>
        <td>4</td>
        <td>存在少数语法错误</th>
        <td>少数语义缺失</td>
    </tr>
    <tr>
        <td>3</td>
        <td>存在部分语法错误</td>
        <td>部分语义缺失或存在少数无关语义</th>
    </tr>
    <tr>
        <td>2</td>
        <td>存在大量语法错误</td>
        <td>大量语义缺失或存在部分无关语义</td>
    </tr>
    <tr>
        <td>1</td>
        <td>不能看出语法结构</td>
        <td>语义完全无关</td>
    </tr>
</table>

## 成绩
<table>
    <tr>
        <th rowspan=2 colspan=2>模型</th>
        <th colspan=2>Java到C#</th>
        <th colspan=2>C#到Java</th>
    </tr>
    <tr>
        <th>语法</th>
        <th>语义</th>
        <th>语法</th>
        <th>语义</th>
    </tr>
    <tr>
        <th rowspan=6>Gate-HPGN</th>
        <td>1</td>
        <td>4.44</td>
        <td>4.47</td>
        <td>4.49</td>
        <td>4.49</td>
    </tr>
    <tr>
        <td>2</td>
        <td>4.16</td>
        <td>4.47</td>
        <td>4.07</td>
        <td>4.38</td>
    </tr>
    <tr>
        <td>3</td>
        <td>4.66</td>
        <td>4.68</td>
        <td>4.73</td>
        <td>4.74</td>
    </tr>
    <tr>
        <td>4</td>
        <td>4.30</td>
        <td>4.34</td>
        <td>4.67</td>
        <td>4.54</td>
    </tr>
    <tr>
        <td>5</td>
        <td>4.45</td>
        <td>4.50</td>
        <td>4.62</td>
        <td>4.53</td>
    </tr>
    <tr>
        <td>平均</td>
        <td>4.40</td>
        <td>4.49</td>
        <td>4.52</td>
        <td>4.54</td>
    </tr>
    <tr>
        <th rowspan=6>Transformer</th>
        <td>1</td>
        <td>4.64</td>
        <td>4.21</td>
        <td>4.73</td>
        <td>4.29</td>
    </tr>
    <tr>
        <td>2</td>
        <td>4.28</td>
        <td>4.14</td>
        <td>4.22</td>
        <td>4.16</td>
    </tr>
    <tr>
        <td>3</td>
        <td>4.79</td>
        <td>4.67</td>
        <td>4.83</td>
        <td>4.69</td>
    </tr>
    <tr>
        <td>4</td>
        <td>4.44</td>
        <td>4.12</td>
        <td>4.78</td>
        <td>4.37</td>
    </tr>
    <tr>
        <td>5</td>
        <td>4.64</td>
        <td>4.45</td>
        <td>4.79</td>
        <td>4.35</td>
    </tr>
    <tr>
        <td>平均</td>
        <td>4.56</td>
        <td>4.32</td>
        <td>4.67</td>
        <td>4.37</td>
    </tr>
</table>

# 输出案例

## 与对比模型对比的案例研究
<table>
    <tr>
        <th colspan=2>样例1</th>
    </tr>
    <tr>
        <td>源代码</td>
        <td><code>public static final WeightedTerm[] getTerms(Query query){return getTerms(query,false);}</code></td>
    </tr>
    <tr>
        <td>参考代码</td>
        <td><code>public static WeightedTerm[] GetTerms(Query query){return GetTerms(query, false);} </code></td>
    </tr>
    <tr>
        <td>tree-to-tree</td>
        <td><code>public static WeightedTerm[] identifier(Query query) { return capacity(query, false); }</code></td>
    </tr>
    <tr>
        <td>Transformer</td>
        <td><code>public static WeightedTerm[] GetTerms(Query queryTerm) { return GetTerms(query, queryTerms, Term); }</code></td>
    </tr>
    <tr>
        <td>Gate-HPGN</td>
        <td><code>public static WeightedTerm[] GetTerms(Query query)) { return GetTerms(query,false); }</code></td>
    </tr>
</table>

<table>
    <tr>
        <th colspan=2>样例2</th>
    </tr>
    <tr>
        <td>源代码</td>
        <td><code>public long skip(long n){int s = (int) Math.min(available(), Math.max(0, n));ptr += s;return s;}</code></td>
    </tr>
    <tr>
        <td>参考代码</td>
        <td><code>public override long Skip(long n){int s = (int)Math.Min(Available(), Math.Max(0, n)); ptr += s;return s;}</code></td>
    </tr>
    <tr>
        <td>tree-to-tree</td>
        <td><code>public override long skip(long n) { int s = (int) Math.Min(0, n); ptr += n; return s; }</code></td>
    </tr>
    <tr>
        <td>Transformer</td>
        <td><code>public override long Skip(long n) { int s = (int)(MIN_SHIFT); return ((long)((ulong)block >> shift)) & unchecked((int)(0xff)); }</code></td>
    </tr>
    <tr>
        <td>Gate-HPGN</td>
        <td><code>public virtual long skip(long n) { int s = (int)(offsetmin(available(), Math.max(0, n)); ptr += s; return s; }</code></td>
    </tr>
</table>

## 与消融实验模型对比的案例研究
<table>
    <tr>
        <th colspan=2>样例3</th>
    </tr>
    <tr>
        <td>源代码</td>
        <td><code>public final boolean hasPassedThroughNonGreedyDecision() { return passedThroughNonGreedyDecision; }</code></td>
    </tr>
    <tr>
        <td>参考代码</td>
        <td><code>public bool hasPassedThroughNonGreedyDecision() { return passedThroughNonGreedyDecision; } </code></td>
    </tr>
    <tr>
        <td>Resnet-HPGN</td>
        <td><code>public bool H hasPassedThroughNonGreedyDecision() { return passedThroughNonGreedyDecision; }</code></td>
    </tr>
    <tr>
        <td>Gate-HPGN</td>
        <td><code>public bool hasPassedThroughNonGreedyDecision() { return passedThroughNonGreedyDecision; }</code></td>
    </tr>
    <tr>
        <td>Base-HPGN</td>
        <td><code>public bool HPasassedThroughNonGreedyDecision() { return passedThroughNonGreedyDecision; }</code></td>
    </tr>
    <tr>
        <td>指针生成网络</td>
        <td><code>public virtual bool hasPassedThroughNonGreedyDecision() { return passedroughNonGreedyDecision; } } } } } } } } } } }</code></td>
    </tr>
</table>
<table>
    <tr>
        <th colspan=2>样例4</th>
    </tr>
    <tr>
        <td>源代码</td>
        <td><code>public UpdateUserRequest(String userName) { setUserName(userName); }</code></td>
    </tr>
    <tr>
        <td>参考代码</td>
        <td><code>public UpdateUserRequest(string userName) { _userName = userName; }</code></td>
    </tr>
    <tr>
        <td>Resnet-HPGN</td>
        <td><code>public UpdateUserRequest(string userName) { _userName =NameName; }</code></td>
    </tr>
    <tr>
        <td>Gate-HPGN</td>
        <td><code>public UpdateUserRequest(string userName) { _userName = userName; }</code></td>
    </tr>
    <tr>
        <td>Base-HPGN</td>
        <td><code>public UpdateUserRequest(string userName) { _userName = groupName; }</code></td>
    </tr>
    <tr>
        <td>指针生成网络</td>
        <td><code>public UpdateUserRequest(string userName) { _UserNameuserName); }</code></td>
    </tr>
</table>

## 与代码迁移工具对比的案例研究
<table>
    <tr>
        <th colspan=2>样例5</th>
    </tr>
    <tr>
        <td>源代码</td>
        <td><code>
        public LatvianStemFilterFactory(Map<String,String> args) { super(args); 
    if (!args.isEmpty()) { throw new IllegalArgumentException("Unknown parameters: " + args); } 
}</code></td>
    </tr>
    <tr>
        <td>参考代码</td>
        <td><code>public LatvianStemFilterFactory(IDictionary<string, string> args): base(args) { 
    if (args.Count > 0) { throw new System.ArgumentException("Unknown parameters: " + args); } 
}</code></td>
    </tr>
    <tr>
        <td>代码迁移工具</td>
        <td><code>using System.Collections.Generic;
//JAVA TO C# CONVERTER WARNING: The following constructor is declared outside of its associated class:
//ORIGINAL LINE: public LatvianStemFilterFactory(Map<String,String> args)
public LatvianStemFilterFactory(IDictionary<string, string> args) : base(args) {
    if (args.Count > 0) {throw new System.ArgumentException("Unknown parameters: " + args);}
}</code></td>
    </tr>
    <tr>
        <td>Gate-HPGN</td>
        <td><code>public LatvianStemFilterFactory(IDictionary<string, string> args): base(args) {
    if (args.Count > 0) {throw new System.ArgumentException("Unknown parameters: " + args);}
}</code></td>
    </tr>
</table>
<table>
    <tr>
        <th colspan=2>样例6</th>
    </tr>
    <tr>
        <td>源代码</td>
        <td><code>public PutIntegrationResult putIntegration(PutIntegrationRequest request) { 
    request = beforeClientExecution(request); 
    return executePutIntegration(request); 
}</code></td>
    </tr>
    <tr>
        <td>参考代码</td>
        <td><code>public virtual PutIntegrationResponse PutIntegration(PutIntegrationRequest request) { 
    var options = new InvokeOptions(); 
    options.RequestMarshaller = PutIntegrationRequestMarshaller.Instance; 
    options.ResponseUnmarshaller = PutIntegrationResponseUnmarshaller.Instance; 
    return Invoke<PutIntegrationResponse>(request, options); 
}</code></td>
    </tr>
    <tr>
        <td>代码迁移工具</td>
        <td><code>public virtual PutIntegrationResult putIntegration(PutIntegrationRequest request) {
    request = beforeClientExecution(request);
    return executePutIntegration(request);
}</code></td>
    </tr>
    <tr>
        <td>Gate-HPGN</td>
        <td><code>public virtual PutIntegrationResponse PutIntegration(PutIntegrationRequest request) { 
    var options = new InvokeOptions(); 
    options.RequestMarshaller = PutIntegrationRequestMarshaller.Instance; 
    options.ResponseUnmarshaller = PutIntegrationResponseUnmarshaller.Instance; 
    return Invoke<PutIntegrationResponse>(request, options); 
}</code></td>
    </tr>
</table>

## HPGN的错误输出案例研究
错误：不符合EM指标评价方法的代码
<table>
    <tr>
        <th colspan=2>样例7</th>
    </tr>
    <tr>
        <td>源代码</td>
        <td><code>public boolean remove(Object o) { synchronized (mutex) { return delegate().remove(o); } }</code></td>
    </tr>
    <tr>
        <td>参考代码</td>
        <td><code>public virtual bool remove(object @object) { lock (mutex) { return c.remove(@object); } }</code></td>
    </tr>
    <tr>
        <td>Gate-HPGN</td>
        <td><code>public override bool Remove(object o) { lock (mutex) { return } }</code></td>
    </tr>
</table>
<table>
    <tr>
        <th colspan=2>样例8</th>
    </tr>
    <tr>
        <td>源代码</td>
        <td><code>public static String toHex(long value) { 
StringBuilder sb = new StringBuilder(16); 
writeHex(sb, value, 16, ""); 
    return sb.toString(); 
}</code></td>
    </tr>
    <tr>
        <td>参考代码</td>
        <td><code>public static string ToHex(int value) { return ToHex((long)value, 8); }</code></td>
    </tr>
    <tr>
        <td>Gate-HPGN</td>
        <td><code>public static string ToHex(long value) { 
StringBuilder sb = new StringBuilder(16); 
writeHex(sb, value, 16, ""); 
return sb.ToString(); 
}</code></td>
    </tr>
</table>