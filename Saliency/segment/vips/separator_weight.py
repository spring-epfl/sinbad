from .weight_rule import WeightRule


class SeparatorWeight:
    def __init__(self, nodeList):
        WeightRule.initialize(nodeList)

    def service(self, separatorList, hrList):
        for sep in separatorList:
            if sep.oneSide == None or sep.otherSide == None:
                continue

            WeightRule.rule1(sep)
            WeightRule.rule2(sep, hrList)
            WeightRule.rule3(sep)
            WeightRule.rule4(sep)
            WeightRule.rule5(sep)
