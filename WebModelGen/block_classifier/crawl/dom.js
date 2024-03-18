function isVisible(elem) {


    // edge case for labels
    if (elem.nodeName == "LABEL"
        && elem.parentNode.querySelector("input")
        && getComputedStyle(elem).position == "absolute"
        && elem.attributes.for.value == elem.parentNode.querySelector('input').id) {
        return isVisible(elem.parentNode.querySelector('input'))
    }

    if (!(elem instanceof Element)) throw Error('DomUtil: elem is not an element.');
    const style = getComputedStyle(elem);
    if (style.display === 'none') return false;
    if (style.visibility !== 'visible') return false;
    if (style.opacity < 0.1) return false;
    if (elem.offsetWidth + elem.offsetHeight + elem.getBoundingClientRect().height +
        elem.getBoundingClientRect().width === 0) {
        return false;
    }
    const elemTopLeft = {
        x: elem.getBoundingClientRect().left,
        y: elem.getBoundingClientRect().top
    };
    if (elemTopLeft.x + elem.offsetWidth < 0) return false;
    if (elemTopLeft.x > (document.documentElement.clientWidth || window.innerWidth)) return false;
    if (elemTopLeft.y + elem.offsetHeight < 0) return false;
    if (elemTopLeft.y > (document.documentElement.clientHeight || window.innerHeight)) return false;

    return true
    // try {
    //     let pointContainer = document.elementFromPoint(elemCenter.x, elemCenter.y);
    //     do {
    //         if (pointContainer === elem) return true;
    //     } while (pointContainer = pointContainer.parentNode);
    //     return false;
    // } catch (e) {
    //     return true;
    // }
}

function inView(elem) {
    if (!elem.getBoundingClientRect) return true

    let left = elem.getBoundingClientRect().left
    let top = elem.getBoundingClientRect().top
    let bottom = elem.getBoundingClientRect().bottom
    let right = elem.getBoundingClientRect().right

    return left <= document.documentElement.clientWidth && top <= document.documentElement.clientHeight && bottom >= 0 && right >= 0
}

function toJSON(node, is_in_view_only = true) {
    node = node || this;
    var obj = {
        nodeType: node.nodeType
    };
    if (node.tagName) {
        obj.tagName = node.tagName.toLowerCase();
    } else
        if (node.nodeName) {
            obj.nodeName = node.nodeName;
        }
    if (node.nodeValue) {
        obj.nodeValue = node.nodeValue;
    }
    if (node.nodeType == 1) {
        obj.visual_cues = getCSS(node);
    }

    var attrs = node.attributes;
    if (attrs) {
        var length = attrs.length;
        let arr = obj.attributes = new Array(length);
        for (var i = 0; i < length; i++) {
            arr[i] = {
                key: attrs[i].name,
                value: attrs[i].nodeValue
            };
        }
    }

    var childNodes = node.childNodes;
    if (childNodes) {
        length = childNodes.length;
        let arr = obj.childNodes = new Array();
        for (i = 0; i < length; i++) {
            if (childNodes[i].tagName != 'script' && (!is_in_view_only || inView(childNodes[i]))) {
                arr.push(toJSON(childNodes[i]));
            }
        }
    }
    return obj;
}

function getCSS(node) {
    var visual_cues = {};
    style = window.getComputedStyle(node);
    visual_cues["bounds"] = node.getBoundingClientRect();
    visual_cues["font-size"] = style.getPropertyValue("font-size");
    visual_cues["font-weight"] = style.getPropertyValue("font-weight");
    visual_cues["background-color"] = style.getPropertyValue("background-color");
    visual_cues["display"] = style.getPropertyValue("display");
    visual_cues["visibility"] = style.getPropertyValue("visibility");
    visual_cues['is_visible'] = isVisible(node)
    visual_cues["text"] = node.innerText
    visual_cues["className"] = node.className
    return visual_cues;
}

function toDOM(obj) {
    if (typeof obj == 'string') {
        obj = JSON.parse(obj);
    }
    var node, nodeType = obj.nodeType;
    switch (nodeType) {
        case 1: //ELEMENT_NODE
            node = document.createElement(obj.tagName);
            var attributes = obj.attributes || [];
            for (var i = 0, len = attributes.length; i < len; i++) {
                var attr = attributes[i];
                node.setAttribute(attr[0], attr[1]);
            }
            break;
        case 3: //TEXT_NODE
            node = document.createTextNode(obj.nodeValue);
            break;
        case 8: //COMMENT_NODE
            node = document.createComment(obj.nodeValue);
            break;
        case 9: //DOCUMENT_NODE
            node = document.implementation.createDocument();
            break;
        case 10: //DOCUMENT_TYPE_NODE
            node = document.implementation.createDocumentType(obj.nodeName);
            break;
        case 11: //DOCUMENT_FRAGMENT_NODE
            node = document.createDocumentFragment();
            break;
        default:
            return node;
    }
    if (nodeType == 1 || nodeType == 11) {
        var childNodes = obj.childNodes || [];
        for (i = 0, len = childNodes.length; i < len; i++) {
            node.appendChild(toDOM(childNodes[i]));
        }
    }
    return node;
}