import json
import argparse
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import RegexpParser
import copy
import os
import sys

# 下载NLTK资源
print("正在下载NLTK资源...")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

def extract_subject_object(sentence):
    """
    从句子中提取主语和宾语词组
    
    参数:
    sentence (str): 输入的句子
    
    返回:
    list: 提取的主语和宾语词组列表
    """
    if not sentence or not isinstance(sentence, str):
        print(f"警告: 无效的句子输入: {sentence}")
        return []
        
    try:
        # 分词和词性标注
        tokens = word_tokenize(sentence)
        tagged = pos_tag(tokens)
        
        # 定义简化的语法规则 - 只识别名词短语
        grammar = """
NP: {<DT>?<JJ.*>*<NN.*>+}
    {<PRP>}
"""
        
        # 创建解析器
        parser = RegexpParser(grammar)
        
        # 解析句子
        parsed = parser.parse(tagged)
        
        # 找出主语和宾语（名词短语）
        noun_phrases = []
        
        # 遍历解析树，提取名词短语
        for subtree in parsed.subtrees():
            if subtree.label() == 'NP':  # 名词短语（可能是主语或宾语）
                phrase = ' '.join(word for word, tag in subtree.leaves())
                noun_phrases.append(phrase)
        
        return noun_phrases
    except Exception as e:
        print(f"提取词组时出错: {e}")
        return []

def process_annotation_json(input_json_path, output_json_path):
    """
    处理输入的annotation_sentence.json并保存为annotation_word.json
    """
    try:
        # 读取输入的JSON文件
        print(f"正在读取 {input_json_path}...")
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 创建输出的JSON结构（复制原始结构）
        output_data = copy.deepcopy(data)
        
        total_items = sum(len(split_data) for split_name, split_data in data.items())
        processed_items = 0
        
        # 遍历所有数据条目
        for split_name, split_data in data.items():
            print(f"处理数据集 '{split_name}' 中的 {len(split_data)} 个条目...")
            
            for i, item in enumerate(split_data):
                # 显示进度
                processed_items += 1
                if processed_items % 100 == 0 or processed_items == total_items:
                    print(f"进度: {processed_items}/{total_items} ({processed_items/total_items*100:.1f}%)")
                
                # 新建一个列表用于存储每个句子提取的词组
                word_phrases = []
                
                # 确保'report_sentences'字段存在
                if 'report_sentences' not in item:
                    print(f"警告: 条目 {item.get('id', i)} 中没有 'report_sentences' 字段，跳过此条目")
                    continue
                
                # 处理每个句子
                for sentence in item['report_sentences']:
                    # 提取主语和宾语
                    phrases = extract_subject_object(sentence)
                    # 将提取的词组添加到列表中
                    word_phrases.extend(phrases)
                
                # 将原始的'report_sentences'替换为提取的词组
                output_data[split_name][i]['report_words'] = word_phrases
                # 删除原始的'report_sentences'字段
                if 'report_sentences' in output_data[split_name][i]:
                    del output_data[split_name][i]['report_sentences']
        
        # 保存输出的JSON文件
        print(f"正在保存结果到 {output_json_path}...")
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        
        return output_data
    except Exception as e:
        print(f"处理JSON文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(
        description='从 annotation_sentence.json 中提取词组，生成 annotation_phrase.json')
    parser.add_argument('--input', dest='input_json_path',
                        default='data/iu_xray/annotation_sentence.json',
                        help='输入的 annotation_sentence.json 路径')
    parser.add_argument('--output', dest='output_json_path',
                        default='data/iu_xray/annotation_phrase.json',
                        help='输出的 annotation_phrase.json 路径')
    args = parser.parse_args()
    input_json_path = args.input_json_path
    output_json_path = args.output_json_path

    # 检查输入文件是否存在
    if not os.path.exists(input_json_path):
        print(f"错误: 找不到输入文件 '{input_json_path}'")
        print("请确保文件路径正确，或修改代码中的文件路径。")
        return
    
    # 处理JSON文件
    output_data = process_annotation_json(input_json_path, output_json_path)
    
    if output_data:
        print(f"处理完成。结果已保存到 {output_json_path}")

# 测试用例
def test_extract():
    """
    测试词组提取功能
    """
    test_sentences = [
        "The heart size and pulmonary vascularity appear within normal limits.",
        "A large hiatal hernia is noted.",
        "The lungs are free of focal airspace disease.",
        "No pneumothorax or pleural effusion is seen.",
        "Degenerative changes are present in the spine."
    ]
    
    print("测试主语和宾语提取功能:")
    for sentence in test_sentences:
        phrases = extract_subject_object(sentence)
        print(f"\n原句: {sentence}")
        print(f"提取的主语和宾语: {phrases}")

if __name__ == "__main__":
    # 取消注释以下行来测试词组提取功能
    #test_extract()
    
    # 运行主程序
    main()