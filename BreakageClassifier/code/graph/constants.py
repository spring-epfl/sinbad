DOM_TYPE = 1
COMMENT_TYPE = 3
INTERACTION_TYPE = 10
ERROR_TYPE = 11
REQUEST_TYPE = 12
SCRIPT_TYPE = 13
SCRIPT_RELATE_DOM_TYPE = 14
INTERACT_RELATE_DOM_TYPE = 15
INTERACT_ERROR_TYPE = 16

JS_ERROR_SYMBOLS = [
    "window.console.error",
    "window.console.warn",
    "window.onerror",
]

JS_QUERY_SELECTOR_ALL = "window.document.querySelectorAll"

JS_QUERY_SELECTOR = "window.document.querySelector"

JS_QUERY_SELECTORS = [JS_QUERY_SELECTOR_ALL, JS_QUERY_SELECTOR]

JS_GET_ELEMENTS_BY_TAG_NAME = "window.document.getElementsByTagName"
JS_GET_ELEMENTS_BY_CLASS_NAME = "window.document.getElementsByClassName"
JS_GET_ELEMENT_BY_ID = "window.document.getElementById"


JS_ELEM_SYMBOLS = [
    JS_QUERY_SELECTOR,
    JS_QUERY_SELECTOR_ALL,
    JS_GET_ELEMENTS_BY_TAG_NAME,
    JS_GET_ELEMENTS_BY_CLASS_NAME,
    JS_GET_ELEMENT_BY_ID,
]
