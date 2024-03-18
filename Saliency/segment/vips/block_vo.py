class BlockVo:

    id = None
    x = 0
    y = 0
    width = 0
    height = 0
    boxs = []
    parent = None
    children = []
    isVisualBlock = True
    isDividable = True
    Doc = 0
    count = 1

    def __init__(self):
        self.id = str(BlockVo.count)
        BlockVo.count += 1
        self.boxs = []
        self.children = []

    def add_box(self, box):
        self.boxs.append(box)

    def refresh(self):

        visible_boxs = [box for box in self.boxs if box.visual_cues["is_visible"]]

        for i, box in enumerate(visible_boxs):
            if i == 0:
                self.x = box.visual_cues["bounds"]["x"]
                self.y = box.visual_cues["bounds"]["y"]
                self.width = box.visual_cues["bounds"]["width"]
                self.height = box.visual_cues["bounds"]["height"]

            else:
                RBX = self.x + self.width
                RBY = self.y + self.height
                boxRBX = (
                    box.visual_cues["bounds"]["x"] + box.visual_cues["bounds"]["width"]
                )
                boxRBY = (
                    box.visual_cues["bounds"]["y"] + box.visual_cues["bounds"]["height"]
                )
                RBX = boxRBX if (boxRBX > RBX) else RBX
                RBY = boxRBY if (boxRBY > RBY) else RBY
                self.x = (
                    box.visual_cues["bounds"]["x"]
                    if (box.visual_cues["bounds"]["x"] < self.x)
                    else self.x
                )
                self.y = (
                    box.visual_cues["bounds"]["y"]
                    if (box.visual_cues["bounds"]["y"] < self.y)
                    else self.y
                )
                self.width = RBX - self.x
                self.height = RBY - self.y

    @staticmethod
    def refreshBlock(block):
        block.refresh()
        for blockVo in block.children:
            BlockVo.refreshBlock(blockVo)
