#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
将文本整合到 train、test、val 三个文件中
"""

"""
import pymysql
import os
_connect = pymysql.connect(
        host='localhost',
        db='test',
        user='root',
        passwd='fb269948985',
        use_unicode=True)
cursor = _connect.cursor()
string = '机关效能 经济管理 安全生产 城乡建设 市场监管 人力资源 社会保障 民政 生态环境 国土资源 教育文化'
string = string.split(' ')
print(string)
for k in string:
 #   fly_sql = '''SELECT * FROM dataset WHERE _type = '%s' LIMIT 100''' % (k)
# fly_sql = '''SELECT * FROM dataset2 WHERE _type = '%s' ''' % (string[-1])
    num = cursor.execute(fly_sql)
    res = cursor.fetchall()
    # path = "C:/迅雷下载/data/%s/" % (k)
    path = 'C:/迅雷下载/data/data/total/'
    i = 0
    for j in res:
        temp = path+k+str(i)+'.txt'
        file = open(temp,'w',encoding='utf-8')
        file.write(j[3])
        file.close()
        i += 1
    for i in res:
        print(i)
# print(res)
# _connect.commit()

_connect.close()
cursor.close()
"""
#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
将文本整合到 train、test、val 三个文件中
"""

import os

def _read_file(filename):
    """读取一个文件并转换为一行"""
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read().replace('\n', '').replace('\t', '').replace('\u3000', '')

def save_file(dirname):
    """
    将多个文件整合并存到3个文件中
    dirname: 原数据目录
    文件内容格式:  类别\t内容
    """
    f_train = open('C:/迅雷下载/data/data/cnews.train.txt', 'w', encoding='utf-8')
    f_test = open('C:/迅雷下载/data/data/cnews.test.txt', 'w', encoding='utf-8')
    f_val = open('C:/迅雷下载/data/data/cnews.val.txt', 'w', encoding='utf-8')
    count = 0
    i,j,k = 1,1,1
    temp = ''
    flag = 0
    for category in os.listdir(dirname):   # 分类目录
        # cat_dir = dirname  + category
        flag = 1
        if flag and (temp[0:2] != category[0:2]):
            i, j, k = 1, 1, 1
        temp = category
        files = os.listdir('C:/迅雷下载/data/data/total')
        t = count % 5
        filename = os.path.join(dirname, category)
        content = _read_file(filename)
        if category[0:2] == '民政':
            category = category[0:2]
        else:
            category = category[0:4]
        if t == 1 or t == 2 or t == 3:
            temk = ''
            if len(str(i))<2:
                temk = '0'
            f_train.write(category + '\t' + content + '\n')
            i = int(i)+1
        elif t == 4:
            temk = '0'
            if len(str(j)) < 2:
                temk = ' '
            f_test.write(category +  '\t' + content + '\n')
            j = int(j)+1
        else:
            temk = '0'
            if len(str(k)) < 2:
                temk = ' '
            f_val.write(category +'\t' + content + '\n')
            k = int(k)+1
        count += 1

        print('Finished:', category)

    f_train.close()
    f_test.close()
    f_val.close()


if __name__ == '__main__':
    save_file('C:/迅雷下载/data/data/total/')
    print(len(open('C:/迅雷下载/data/data/cnews.train.txt', 'r', encoding='utf-8').readlines()))
    print(len(open('C:/迅雷下载/data/data/cnews.test.txt', 'r', encoding='utf-8').readlines()))
    print(len(open('C:/迅雷下载/data/data/cnews.val.txt', 'r', encoding='utf-8').readlines()))
