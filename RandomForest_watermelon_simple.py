import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import random
from pprint import pprint

# claculate original_Entropty
def cal_original_Entropty(df):
    n = df.shape[0] #行数
    label_info = df.iloc[:, -1].value_counts().to_dict()     #对最后一列值 和 出现次数做统计
    label_name = label_info.keys() 
    #print("label_name",label_name)#dict.keys
    label_frequency = label_info.values() #dict.values
    original_Entropy =0
    for frequency in label_frequency:
        p = frequency/n
        entropy = -p*np.log2(p)
        original_Entropy += entropy
    return original_Entropy


def cal_each_Entropy(df):
    original_Entropy=cal_original_Entropty(df)
    n = df.shape[0] #行数
    column_name_list=df.columns.tolist()
    entropy_list_for_characters=[]
    info_gain_list=[]  
    for i in range(df.shape[1]-1):
        #print("For character:{}". format( column_name_list[i] ))
        column_name =df.columns[i] #当前列的列名
        subClass_name= df.loc[:, column_name].value_counts().index.tolist()   #当前列的，子特征名(不是以当前子df的去算，而是全部
        subClass_frequency = df.iloc[:, i].value_counts().values.tolist()#子特征对应个数
        subClass_count=sum(subClass_frequency)# 第1列一共有多少行
        classification_list= [] #分类
        classification_list=df.iloc[:, -1].value_counts().index.tolist() #类名
        subClass_label_List = []
        entropy_of_subClass=[]          
        subclass_dict={}
        each_subClass_entropy_list=[]
        for subClass in subClass_name:#(子特征1，0)
            #print("\tsubClass",subClass)
            entropy_of_subClass=[]
            for classcification in classification_list: #每个子标签所对应的分类，几个no,几个yes
                a= df[(df.iloc[:,i]==subClass)&(df.iloc[:,-1].values==classcification)] #返回符合条件的行，和具体值
                a=a.shape[0] #当前标签的个数
                subclass_dict.update({classcification:a}) #子特征①的对应标签数
            subClass_label_List= list(subclass_dict.values())
            subClass_total=sum(list(subclass_dict.values())) #子特征①个数
            total_entropy = 0
            subClass_entropy=0
            little_entropy=0
           # little_entropy=0           
            #print("subClass_label_List",subClass_label_List)
            for value in subClass_label_List: #每个子分类的ent
                if value ==0:
                    little_entropy=0
                else:
                    p=value/subClass_total
                    sub_entropy=-p*np.log2(p)
                    #print("sub_entropy",sub_entropy)
                    little_entropy+= sub_entropy
            P=subClass_total/subClass_count
            each_subClass_entropy=P*little_entropy #P*
            each_subClass_entropy_list.append(each_subClass_entropy)
            subClass_entropy += each_subClass_entropy #第一列的总entropy
        characters_entropy=sum(each_subClass_entropy_list) #第一列的总entropy
        #entropy_list_for_characters.update({column_name_list[i]:characters_entropy})# 每一列的entropy
        entropy_list_for_characters.append(characters_entropy)
        #print("     entropy_list_for_characters",entropy_list_for_characters,"\n")
    for values in entropy_list_for_characters:
        info_gain= original_Entropy - values     
        info_gain_list.append(info_gain)
    #print("info_gain_list",info_gain_list)
    max_gain= max(info_gain_list)    
    max_gain_index= info_gain_list.index(max(info_gain_list))
    column_name = df.columns.values.tolist()
    max_gain_name= column_name[max_gain_index]        
    return(max_gain_index)

#根据给定列，划分数据集
def split(df, max_gain_index, value):#根节点所对应的子特征，返回的是数据集
    re_df =df[df.iloc[:,max_gain_index]==value].drop(columns=df.columns[[max_gain_index]]) #这种写法也行 
    return re_df 


def createTree(df):

    print( "\tdf", df)
    #random_character = random_character(df, character_size )
    feature_list= df.columns.tolist() #特征名list
    class_list= df.iloc[:,-1].value_counts() #标签信息，种类和个数
    print( "\tfeature_list{}\n  class_list{}".format(feature_list, class_list ))
    if class_list[0] == df.shape[0] or df.shape[1] == 1: #class_list[0] 为数量最多的标签种类
        print("class_list[0]",class_list[0])
        return class_list.index[0] #返回标签名, 整个函数结束
    max_gain_index= cal_each_Entropy(df) #根节点的索引
    best_feature_name = feature_list[max_gain_index] #根节点的名字
    print("best_feature_name",best_feature_name)
    Tree = {best_feature_name:{}}
    del feature_list[max_gain_index]# 把根节点放进字典，之后把它从备选特征中删掉
    value_list = df.iloc[:, max_gain_index].value_counts().index.tolist() #根节点的子特征
    #对每个属性递归建树:
    for subClass in value_list:
        new_dataset = split(df, max_gain_index, subClass)
        #print("\t\tnew_df\n ", new_df , "\n\n")
        Tree[best_feature_name][subClass] = createTree(new_dataset)
    return Tree

def classify(tree, characters, test_Line): #对单行数据进行测试
    #if type(tree) != dict:
    #    tree =  createTree(df
    label = df.iloc[:,-1].unique()
    u = random.randint(0, len(label)-1) #如果一个值没在字典中， 就随机返回一个标签
    classLabel =label[u]
    #print("\ntree", tree)
    #print("characters", characters)
    #print("test_Line", test_Line)
    first_node = next(iter(tree))
    #print("first_node", first_node)
    
    secondDict = tree[first_node] #返回第一个节点的列的索引
    #print("secondDict", secondDict)
    feature_index = characters.index(first_node)
    for key in secondDict.keys():
        if test_Line[feature_index] == key :
            if type(secondDict[key] )==dict:
                classLabel = classify(secondDict[key], characters, test_Line)
            else:
                classLabel = secondDict[key]
    return classLabel

def accuracy_classify(tree, test): #测试集，准确率计算

    characters = list(test.columns) 
    result = []
    for i in range(test.shape[0]):
        test_Line = test.iloc[i, :-1]
        classLabel = classify(tree, characters, test_Line) # 对每一个值做预测，返回标签
       # print("classLabel ", classLabel)
        result.append(classLabel)
    return result



# 切分训练集和测试集
def train_test_split(df, test_size):
    if isinstance(test_size, float):  #可输入百分数，也可整数 (isinstance是判断类型的函数)
        test_size = round(test_size * len(df))
    index = df.index.tolist()
    test_index = random.sample(population = index, k = test_size) #
    #test_index = [0,10,1]
    #print("test_index",test_index)
    test_df = df.loc[test_index]
    train_df = df.drop(test_index)
    return train_df , test_df


# 随机抽样
def random_sampling(train, sample_size):
    train.index = range(len(train))
    train_size = train.shape[0]
    sample_index = np.random.randint(low = 0, high = len(train.index), size = sample_size ) #用于生成一个指定范围内的整数
    random_sampled_train = train.iloc[sample_index]
    #print("random_sampled_train\n{}\n".format(random_sampled_train))
    return random_sampled_train


# 随机选取特征, 指定要除掉的个数， 剩下的构成新的df
def random_character(dataset, del_character_size):
    #column_list  =list(range(random_sampled_train.shape[1] -1)) # 除了最后一列
    column_name = list(dataset.columns) 
    del(column_name[-1])                    #除了最后一列
    drop_character_name = random.sample(population = column_name, k = del_character_size)
    new_train_df = dataset.drop(columns= drop_character_name)
    #print("new_train_df\n{}\n".format(new_train_df))
    return new_train_df            # 构建一个新的df

#建森林
def random_forest_algorithm(df, n_trees, sample_size, del_character_size):
    #train_df, n_trees, n_boostrap, n_features, tree_max_depth
    forest=[]
    for i in range(n_trees):
        random_sampled_train = random_sampling(df, sample_size)
        new_train_df  = random_character(random_sampled_train, del_character_size)
        tree = createTree(new_train_df)
        forest.append(tree)
    pprint(forest, width = 60)
    return forest

"""
def random_forest_predictions(test, forest):
        prediction_results = {}
        for i in range(len(forest)):
            column_name = "tree_{}".format(i)
            prediction = decision_tree_predictions(test, tree = forest[i])
            prediction_results[column_name] = prediction
        
        column_name = list(test.columns)
        print(prediction_results)
        prediction_results = pd.DataFrame.from_dict(prediction_results, orient ="index", columns= column_name)
        random_forest_predictions = prediction_results.mode(axis = 1)[0]
        return random_forest_predictions
"""

def calculate_accuracy( predictions, labels):
        predictions_correct = predictions == labels
        accuracy = predictions_correct.mean()
        print("\naccuracy", accuracy)
        return  accuracy

    
def random_forest_predictions(test, forest):
    prediction = [] 
    column_name= []
    for i in range(len(forest)):
        predicte = accuracy_classify(tree = forest[i],  test= test)
        prediction.append(predicte)
        name = "tree_{}".format(i)
        column_name.append(name)

    predict = prediction
    index_row = column_name
    index_columns= list(test.index)
    prediction_df = pd.DataFrame(prediction)

    result =pd.DataFrame(predict, columns = index_columns, index = index_row)
    result_df = pd.DataFrame(result.values.T, columns=index_row, index= index_columns)

    random_forest_predictions = result_df.mode(axis = 1)[0] 

    return random_forest_predictions


#if __name__ == "__main__":
path="D:/Data/Dataset/watermelon.csv"
df = pd.read_csv(path)
#print(df)
df = df.drop(["No.", "Density", "Sugar"], axis=1)
df= df.rename(columns={"melon":"label"})
 
train, test = train_test_split(df, test_size = 4)
print(train,"\n", test)

# Random forestを作る
forest = random_forest_algorithm(train, n_trees = 30, sample_size =10 , del_character_size = 1)
print(len(forest))
predictions = random_forest_predictions(test, forest )  # 用森林做预测，对结果取第一号众数，众数为预测结果
accuracy = calculate_accuracy(predictions, test.label) 



"""
prediction= []
#for i in range(len(forest)):
for tree in forest:
      a= accuracy_classify(tree, test)
      prediction.append(a)
#print(prediction)

column_name= []
for i in range(len(forest)):
    name = "tree_{}".format(i)
    column_name.append(name )
#print(column_name)

predict = prediction
index_row = column_name
index_columns= list(test.index)

#各木の予測結果をDataframeに入れる
result =pd.DataFrame(predict, columns = index_columns, index = index_row)
result_df = pd.DataFrame(result.values.T, columns=index_row, index= index_columns)
print("result_df\n{}".format(result_df))

# 多数決で最終予測結果とする
predictions = result_df.mode(axis = 1)[0]
print(predictions)

# 予測の精度を計算
accuracy = calculate_accuracy(predictions, test.label)

"""





