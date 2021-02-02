import numpy as np
class Node():
    def __init__(self,lchild,rchild,value):
        self.lchild=lchild#节点的左子树
        self.rchild=rchild#节点的右子树
        self.value=value#节点的数值
        #self.split_dim=split_dim#用来做划分的维度

class KDTree():
    def __init__(self,data):
        self.dims=len(data[0])#总特征数
        
    def create_kdtree(self,current_data,split_dim):
        #设置递归出口：当全部样本划分完毕时就退出
        if len(current_data)==0:
            return None
        
        mid=self.cal_current_medium(current_data)#计算中位数所在下标
        data_sorted=sorted(current_data,key=lambda x:x[split_dim])#按照切分维度从小到大排序

        #下面三句代码本质上就是二叉树的后序遍历
        lchild=self.create_kdtree(data_sorted[0:mid],self.cal_split_dim(split_dim))#递归地构造左子树
        rchild=self.create_kdtree(data_sorted[mid+1:],self.cal_split_dim(split_dim))#递归地构造右子树
        return Node(lchild,rchild,data_sorted[mid])#连接从根节点出发的左右子树，并返回
    
    #计算下一个划分维度
    def cal_split_dim(self,split_dim):
        return (split_dim+1) % self.dims
    
    #计算当前维度中位数所在下标
    def cal_current_medium(self,current_data):
        return len(current_data)//2

    #搜索element的最近邻点
    def search(self,element):
        
        
dataset = np.array([[2,3],[4,7],[5,4],[7,2],[8,1],[9,6]])#构建训练数据集
kdtree = KDTree(dataset).create_kdtree(dataset,0)#创建KD树,以特征的第0个维度开始做划分
