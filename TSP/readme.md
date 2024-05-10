# 轮盘赌算法原理
轮盘赌选择法（roulette wheel selection）是最简单也是最常用的选择方法，在该方法中，各个个体的选择概率和其适应度值成比例，适应度越大，选中概率也越大。但实际在进行轮盘赌选择时个体的选择往往不是依据个体的选择概率，而是根据**“累积概率”**来进行选择。

## 轮盘赌选择法的过程如下：
（1）计算每个个体的被选中概率p(xi)
（2）计算每个部分的累积概率q(xi)
（3）随机生成一个数组m，数组中的元素取值范围在0和1之间，并将其按从小到大的方式进行排序。若累积概率q(xi)大于数组中的元素m[i]，则个体x(i)被选中，若小于m[i]，则比较下一个个体x(i+1)直至选出一个个体为止。
（4）若需要转中N个个体，则将步骤（3）重复N次即可。
```python
import random
def roulette(select_list):
    sum_val = sum(select_list)
    random_val = random.random()
    probability = 0#累计概率
    for i in range(len(select_list)):
        probability += select_list[i] / sum_val#加上该个体的选中概率
        if probability >= random_val:
            return i#返回被选中的下标
        else:
            continue


if __name__ == '__main__':
    select_list = [10,10,10,10,10,10]
    for i in range(50):
        re = roulette(select_list)
        select_list[re] -= 1#被选中的下标的值减1
        print(select_list)
```

原文链接：https://blog.csdn.net/doubi1/article/details/115923275