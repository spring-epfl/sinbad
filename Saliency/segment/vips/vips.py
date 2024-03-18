from email.errors import NonPrintableDefect
import functools
import logging
from typing import List

from .block_extraction import BlockExtraction
from .block_vo import BlockVo
from .content_structure_construction import ContentStructureConstruction
from .dom_node import DomNode
from .separator_vo import SeparatorVo
from .separator_detection import SeparatorDetection
from .separator_weight import SeparatorWeight


class Vips:
    PDoc = 1
    Round = 1
    url = None
    fileName = None
    browser = None
    count = 0
    imgOut = None
    html = None
    cssBoxList = dict()
    node_list = []
    browser_context = None
    count3 = 0

    def __init__(self, node_list: List[DomNode], browser_context, round=10):

        self.node_list = node_list
        self.browser_context = browser_context
        self.Round = round
        self.logger = logging.getLogger("openwpm")

    def service(self):
        # print(
        #     "-----------------------------Block Extraction------------------------------------"
        # )
        be = BlockExtraction()
        block = be.service(self.node_list)
        block_list: List[BlockVo] = be.blockList
        
        i = 0
        while self.checkDoc(block_list) and i < self.Round:
            # print("blockList.size::", len(block_list))

            # self.browser.get_window_size()['width'], self.browser.get_window_size()['height'])
            sd = SeparatorDetection(
                self.browser_context.window_size["width"],
                self.browser_context.window_size["height"],
            )
            verticalList = []
            verticalList.extend(sd.service(block_list, SeparatorVo.TYPE_VERTICAL))
            # self.imgOut.outSeparator(verticalList, self.fileName, "_vertica_", i)

            horizList = []
            horizList.extend(sd.service(block_list, SeparatorVo.TYPE_HORIZ))
            # self.imgOut.outSeparator(horizList, self.fileName, "_horizontal_", i)

            # print(
            #     "-----------------------Setting Weights for Separators----------------------------"
            #     + str(i)
            # )
            
            hrList = be.hrList
            sw = SeparatorWeight(self.node_list)
            sw.service(horizList, hrList)
            sw.service(verticalList, hrList)

            # print(
            #     "-----------------------Content Structure Construction----------------------------"
            #     + str(i)
            # )
            
            sepList = []
            sepList.extend(horizList)
            sepList.extend(verticalList)
            sepList.sort(key=functools.cmp_to_key(Vips.sepCompare))
            tempList = block_list
            csc = ContentStructureConstruction()
            csc.service(sepList, block)
            BlockVo.refreshBlock(block)
            block_list.clear()
            be.filList(block)
            block_list = be.blockList

            for newBlock in block_list:
                for oldBlock in tempList:
                    if newBlock.id == oldBlock.id:
                        block_list.remove(newBlock)
                        break

            i += 1

        for b in block_list:

            for i, box in enumerate(b.boxs):
                if box.nodeType == 3:
                    b.boxs[i] = box.parentNode

                # check if box is the instance of an interactable item

                parent = box
                while parent is not None:
                    if parent.nodeName in ["button", "a", "iframe", "embed", "video"]:
                        b.boxs[i] = parent
                        break
                    try:
                        parent = parent.parentNode
                    except:
                        parent = None

            b.refresh()

        return block_list

    def checkDoc(self, blocks):
        for blockVo in blocks:
            if blockVo.Doc < self.PDoc:
                return True
        return False

    def setRound(self, round):
        self.Round = round

    @staticmethod
    def sepCompare(sep1, sep2):
        if sep1.compareTo(sep2) < 0:
            return -1
        elif sep1.compareTo(sep2) > 0:
            return 1
        else:
            return 0
