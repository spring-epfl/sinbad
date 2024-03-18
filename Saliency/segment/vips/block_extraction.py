from .block_rule import BlockRule
from .block_vo import BlockVo


class BlockExtraction:

    html = None
    blockList = []
    hrList = []
    cssBoxList = dict()
    block = None
    count = 0
    count1 = 0
    count2 = 0
    count3 = 1
    all_text_nodes = []

    def __init__(self):
        self.block = BlockVo()

    def service(self, nodeList):
        BlockRule.initialize(nodeList)
        body = nodeList[0]
        self.initBlock(body, self.block)
        # print("-----Done Initialization-----")
        self.count3 = 0

        self.dividBlock(self.block)
        # print(self.count2)
        # print("-----Done Division-----")

        BlockVo.refreshBlock(self.block)
        # print("-----Done Refreshing-----")
        self.filList(self.block)
        # print("-----Done Filling-----")
        # self.checkText()
        return self.block

    def initBlock(self, box, block: BlockVo):
        block.add_box(box)
        # print(self.count, "####Here Name=", box.nodeName)
        self.count += 1

        if box.nodeName == "hr":
            self.hrList.append(block)
            self.count1 = 0
        if box.nodeType != 3:
            subBoxList = box.childNodes
            for b in subBoxList:
                try:
                    if (
                        b.nodeName != "script"
                        and b.nodeName != "noscript"
                        and b.nodeName != "style"
                    ):
                        # print(self.count1," : ",b.nodeName,", ",box.nodeName)
                        self.count1 += 1
                        bVo = BlockVo()
                        bVo.parent = block
                        block.children.append(bVo)
                        self.initBlock(b, bVo)
                except AttributeError:
                    # print("Attribute error")
                    pass

    def dividBlock(self, block: BlockVo):
        self.count2 += 1
        
        if block.isDividable and BlockRule.dividable(block):
            block.isVisualBlock = False
            for b in block.children:
                self.count3 += 1
                # print(self.count3)
                self.dividBlock(b)

    def filList(self, block):
        if block.isVisualBlock:
            self.blockList.append(block)
        else:
            for blockVo in block.children:
                self.filList(blockVo)

    def checkText(self):
        for blockVo in self.blockList:
            removed = True
            for box in blockVo.boxs:
                if box.nodeType == 3:
                    if (
                        box.parentNode.nodeName != "script"
                        and box.parentNode.nodeName != "noscript"
                        and box.parentNode.nodeName != "style"
                    ):
                        if not box.nodeValue.isspace() or box.nodeValue == None:
                            removed = False
            if removed:
                self.blockList.remove(blockVo)
