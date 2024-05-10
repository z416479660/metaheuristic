# -*- coding: utf-8 -*-
# reference: https://github.com/Greatpanc/-TSP2-/blob/master/GA_TSP_Main.py
import random


class GAList(object):
    """
    遗传算法对象
    """

    def __init__(self, aCrossRate, aMutationRage, aUnitCount, aGeneLenght,
                 aMatchFun=lambda: 1):
        """ 构造函数 """
        self.crossRate = aCrossRate  # 交叉概率 #
        self.mutationRate = aMutationRage  # 突变概率 #
        self.unitCount = aUnitCount  # 个体数 #
        self.geneLenght = aGeneLenght  # 基因长度 #
        self.matchFun = aMatchFun  # 适配函数
        self.population = []  # 种群
        self.best = None  # 保存这一代中最好的个体
        self.generation = 1  # 第几代 #
        self.crossCount = 0  # 交叉数量 #
        self.mutationCount = 0  # 突变个数 #
        self.bounds = 0.0  # 适配值之和，用于选择时计算概率
        self.initPopulation()  # 初始化种群 #

    def initPopulation(self):
        """
        种群初始化
        :return:
        """
        self.population = []
        unitCount = self.unitCount
        while unitCount > 0:
            gene = [x for x in range(self.geneLenght)]
            random.shuffle(gene)  # 随机洗牌 #
            unit = GAUnit(gene)
            self.population.append(unit)
            unitCount -= 1

    def judge(self):
        """
            函数名：judge(self)
            函数功能：	重新计算每一个个体单元的适配值
                输入	1 	self：类自身
                输出	1	无
            其他说明：无
        """
        self.bounds = 0.0
        self.best = self.population[0]
        for unit in self.population:
            unit.value = self.matchFun(unit)
            self.bounds += unit.value
            if self.best.value < unit.value:  # score为距离的倒数 所以越小越好 #
                self.best = unit

    def cross(self, parent1, parent2):
        """
        交叉算子 根据parent1和parent2基于序列,随机选取长度为n的片段进行交换(n=index2-index1)
        其他说明：进行交叉时采用的策略是,将parent2的基因段tempGene保存下来,然后对基因1所有序列号g依次进行判断,
        如果g在tempGene内,则舍去,否则就保存下来,并在第index1的位置插入tempGene
        :param parent1: 进行交叉的双亲1
        :param parent2: 进行交叉的双亲2
        :return:
        ewGene： 通过交叉后的一个新的遗传个体的基因序列号
        """
        index1 = random.randint(0, self.geneLenght - 1)  # 随机生成突变起始位置 #
        index2 = random.randint(index1, self.geneLenght - 1)  # 随机生成突变终止位置 #
        tempGene = parent2.gene[index1:index2]  # 交叉的基因片段
        newGene = []
        p1len = 0
        for g in parent1.gene:
            if p1len == index1:
                newGene.extend(tempGene)  # 插入基因片段
                p1len += 1
            if g not in tempGene:
                newGene.append(g)
                p1len += 1
        self.crossCount += 1
        return newGene

    def mutation(self, gene):
        """
        对输入的gene个体进行变异,也就是随机交换两个位置的基因号
        :param gene: 进行变异的个体基因序列号
        :return:
        newGene： 通过交叉后的一个新的遗传个体的基因序列
        """
        index1 = random.randint(0, self.geneLenght - 1)
        index2 = random.randint(0, self.geneLenght - 1)
        # 随机选择两个位置的基因交换--变异
        newGene = gene[:]  # 产生一个新的基因序列，以免变异的时候影响父种群
        newGene[index1], newGene[index2] = newGene[index2], newGene[index1]
        self.mutationCount += 1
        return newGene

    def getOneUnit(self):
        """
        通过轮盘赌法,依据个体适应度大小,随机选择一个个体   self.bounds是适应度之和
        :return:
        unit：所选择的个体
        """
        r = random.uniform(0, self.bounds)
        for unit in self.population:
            r -= unit.value
            if r <= 0:
                return unit

        raise Exception("选择错误", self.bounds)

    def newChild(self):
        """
        产生新的子代个体
        1. 轮盘赌选择新个体
        2. 交叉
        3. 变异
        :return:
        GAUnit(gene)：所产生的后代
        """
        parent1 = self.getOneUnit() #轮盘赌选择一个父亲1
        rate = random.random()

        # 按概率交叉
        if rate < self.crossRate:  # 交叉
            parent2 = self.getOneUnit()
            gene = self.cross(parent1, parent2)#轮盘赌选择一个父亲2,对父亲1进行变异
        else:  # 不交叉
            gene = parent1.gene

        # 按概率突变
        rate = random.random()
        if rate < self.mutationRate:
            gene = self.mutation(gene)

        return GAUnit(gene)

    def nextGeneration(self):
        """
            函数名：nextGeneration(self)
            函数功能：	产生下一代
                输入	1 	self：类自身
                输出	1	无
            其他说明：无
        """
        self.judge()
        newPopulation = []  # 新种群
        newPopulation.append(self.best)  # 把最好的个体加入下一代 #
        while len(newPopulation) < self.unitCount:
            newPopulation.append(self.newChild()) # 进行选择、交叉、变异操作
        self.population = newPopulation
        self.generation += 1


class GAUnit(object):
    """
        类名：GAUnit
        类说明：	遗传算法个体类
    """

    def __init__(self, aGene=None, SCORE_NONE=-1):
        """ 构造函数 """
        self.gene = aGene  # 个体的基因序列
        self.value = SCORE_NONE  # 初始化适配值




from tools import GetData, ResultShow, draw


class TSP(object):
    def __init__(self, Position, Dist, CityNum):
        """ 构造函数 """
        self.citys = Position  # 城市坐标
        self.dist = Dist  # 城市距离矩阵
        self.citynum = CityNum  # 城市数量
        # 初始化种群
        self.ga = GAList(aCrossRate=0.7,  # 交叉率
                         aMutationRage=0.02,  # 突变概率
                         aUnitCount=150,  # 一个种群中的个体数
                         aGeneLenght=self.citynum,  # 基因长度（城市数）
                         aMatchFun=self.matchFun())  # 适配函数  what to do

    def distance(self, path):
        """
        求路径长度
        :param path: 路径序列
        :return:
        """
        # 计算从初始城市到最后一个城市的路程
        distance = sum([self.dist[city1][city2] for city1, city2 in
                        zip(path[:self.citynum], path[1:self.citynum + 1])])
        # 计算从初始城市到最后一个城市再回到初始城市所经历的总距离
        distance += self.dist[path[-1]][0]

        return distance

    def matchFun(self):
        """
        :usage:
            class MyOrganism:
                def __init__(self, gene):
                    self.gene = gene

                def distance(self, other_gene):
                    # 这里只是一个示例，您需要根据实际情况计算基因之间的距离
                    return sum(1 for x, y in zip(self.gene, other_gene) if x != y)

            class MySelector:
                def __init__(self, gene):
                    self.gene = gene

                def matchFun(self):
                    return lambda life: 1.0 / self.distance(life.gene)

            # 创建基因模板和个体
            gene_template = [1, 0, 1, 1]
            individual = MyOrganism([1, 1, 0, 0])

            # 创建选择器实例
            selector = MySelector(gene_template)

            # 获取适配函数
            adapt_function = selector.matchFun()

            # 计算个体的适配值
            fitness_value = adapt_function(individual)
            print(f"The fitness value of the individual is: {fitness_value}")
        :return:
        """
        return lambda life: 1.0 / self.distance(life.gene)

    def run(self, generate=0):
        """

        :param generate: 种群迭代的代数
        :return:
        1	distance:最短距离
        2	self.ga.best.gene：最好路径
        3	distance_list：每一代的最好路径列表
        """
        distance_list = []

        while generate > 0:
            self.ga.nextGeneration()
            distance = self.distance(self.ga.best.gene)
            distance_list.append(distance)
            generate -= 1

        return distance, self.ga.best.gene, distance_list


##############################程序入口#########################################
if __name__ == '__main__':
    Position, CityNum, Dist = GetData("../data/TSP25cities.tsp")
    tsp = TSP(Position, Dist, CityNum)
    generate = 1000
    Min_Path, BestPath, distance_list = tsp.run(generate)

    # 结果打印
    BestPath.append(BestPath[0])
    ResultShow(Min_Path, BestPath, CityNum, "GA")
    draw(BestPath, Position, "GA")# True, range(generate), distance_list)
